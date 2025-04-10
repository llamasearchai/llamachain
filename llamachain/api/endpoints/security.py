from fastapi import APIRouter, Depends, HTTPException, Body, File, UploadFile, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import logging
import tempfile
import os
from datetime import datetime
import json

from llamachain.db.session import get_db
from llamachain.db import models
from llamachain.security.auditor import ContractAuditor
from llamachain.security.zk import ZkVerifier
from llamachain.ml.models.vulnerability import VulnerabilityDetectionModel


router = APIRouter()
logger = logging.getLogger(__name__)


class AuditError(Exception):
    """Exception raised for errors during contract auditing."""
    pass


@router.post("/audit/contract")
async def audit_contract(
    contract_file: UploadFile = File(..., description="Smart contract file to audit"),
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    use_ml: bool = Query(True, description="Use machine learning detection"),
    db: AsyncSession = Depends(get_db)
):
    """
    Audit a smart contract for security vulnerabilities.
    
    Uploads and analyzes a smart contract file, detecting common vulnerabilities 
    and security issues using both static analysis and machine learning.
    """
    try:
        # Create temporary file to save the uploaded contract
        with tempfile.NamedTemporaryFile(suffix=".sol" if chain == "ethereum" else ".rs", delete=False) as temp_file:
            # Write file content
            content = await contract_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create contract auditor
            auditor = ContractAuditor()
            
            # Analyze contract
            if use_ml:
                # Load ML model for vulnerability detection
                ml_model = VulnerabilityDetectionModel()
                result = await auditor.analyze_contract_with_ml(temp_file_path, ml_model)
            else:
                # Use only static analysis
                result = await auditor.analyze_contract(temp_file_path)
            
            # Check for audit success
            if not result.get("success", False):
                raise AuditError(result.get("error", "Unknown error during audit"))
            
            # Convert datetime objects to timestamps for JSON serialization
            if "timestamp" in result and isinstance(result["timestamp"], datetime):
                result["timestamp"] = result["timestamp"].timestamp()
            
            # Store audit result in database if contract address is provided
            contract_address = result.get("contract_address")
            if contract_address:
                # Check if contract exists in database
                query = (
                    db.query(models.Contract)
                    .filter(models.Contract.blockchain == models.BlockchainEnum(chain.lower()))
                    .filter(models.Contract.address == contract_address)
                )
                db_contract = await db.execute(query)
                db_contract = db_contract.scalar_one_or_none()
                
                if not db_contract:
                    # Create new contract record
                    db_contract = models.Contract(
                        blockchain=models.BlockchainEnum(chain.lower()),
                        address=contract_address,
                        name=result.get("contract_name", ""),
                        source_code=content.decode("utf-8", errors="ignore"),
                        is_verified=True
                    )
                    db.add(db_contract)
                    await db.flush()
                
                # Create audit record
                audit = models.Audit(
                    contract_id=db_contract.id,
                    timestamp=datetime.utcnow(),
                    auditor="LlamaChain",
                    score=result.get("score", 0),
                    report_data=result
                )
                db.add(audit)
                
                # Create vulnerability records
                for finding in result.get("findings", []):
                    vulnerability = models.Vulnerability(
                        contract_id=db_contract.id,
                        title=finding.get("title", "Unknown"),
                        description=finding.get("description", ""),
                        severity=models.VulnerabilitySeverity(finding.get("severity", "medium").lower()),
                        type=finding.get("type", "Unknown"),
                        line_number=finding.get("line", 0),
                        function_name=finding.get("function", ""),
                        detector=finding.get("detector", "static"),
                        confidence=finding.get("confidence", 1.0),
                        details=finding,
                        is_fixed=False
                    )
                    db.add(vulnerability)
                
                await db.commit()
            
            return result
            
        except AuditError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error auditing contract: {e}")
            raise HTTPException(status_code=500, detail=f"Error auditing contract: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.post("/audit/contract/address")
async def audit_contract_by_address(
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    address: str = Body(..., description="Contract address to audit"),
    use_ml: bool = Query(True, description="Use machine learning detection"),
    db: AsyncSession = Depends(get_db)
):
    """
    Audit a deployed smart contract by its address.
    
    Fetches and analyzes a deployed smart contract, detecting common vulnerabilities 
    and security issues using both static analysis and machine learning.
    """
    try:
        # Create contract auditor
        auditor = ContractAuditor()
        
        # Analyze contract by address
        if use_ml:
            # Load ML model for vulnerability detection
            ml_model = VulnerabilityDetectionModel()
            result = await auditor.analyze_contract_address_with_ml(chain, address, ml_model)
        else:
            # Use only static analysis
            result = await auditor.analyze_contract_address(chain, address)
        
        # Check for audit success
        if not result.get("success", False):
            raise AuditError(result.get("error", "Unknown error during audit"))
        
        # Convert datetime objects to timestamps for JSON serialization
        if "timestamp" in result and isinstance(result["timestamp"], datetime):
            result["timestamp"] = result["timestamp"].timestamp()
        
        # Store audit result in database
        # Check if contract exists in database
        query = (
            db.query(models.Contract)
            .filter(models.Contract.blockchain == models.BlockchainEnum(chain.lower()))
            .filter(models.Contract.address == address)
        )
        db_contract = await db.execute(query)
        db_contract = db_contract.scalar_one_or_none()
        
        if not db_contract:
            # Create new contract record
            db_contract = models.Contract(
                blockchain=models.BlockchainEnum(chain.lower()),
                address=address,
                name=result.get("contract_name", ""),
                source_code=result.get("source_code", ""),
                bytecode=result.get("bytecode", ""),
                is_verified=True
            )
            db.add(db_contract)
            await db.flush()
        
        # Create audit record
        audit = models.Audit(
            contract_id=db_contract.id,
            timestamp=datetime.utcnow(),
            auditor="LlamaChain",
            score=result.get("score", 0),
            report_data=result
        )
        db.add(audit)
        
        # Create vulnerability records
        for finding in result.get("findings", []):
            vulnerability = models.Vulnerability(
                contract_id=db_contract.id,
                title=finding.get("title", "Unknown"),
                description=finding.get("description", ""),
                severity=models.VulnerabilitySeverity(finding.get("severity", "medium").lower()),
                type=finding.get("type", "Unknown"),
                line_number=finding.get("line", 0),
                function_name=finding.get("function", ""),
                detector=finding.get("detector", "static"),
                confidence=finding.get("confidence", 1.0),
                details=finding,
                is_fixed=False
            )
            db.add(vulnerability)
        
        await db.commit()
        
        return result
        
    except AuditError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error auditing contract: {e}")
        raise HTTPException(status_code=500, detail=f"Error auditing contract: {str(e)}")


@router.get("/vulnerabilities/{chain}/{address}")
async def get_contract_vulnerabilities(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Contract address"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get known vulnerabilities for a smart contract.
    
    Retrieves all detected vulnerabilities for a specific contract address.
    """
    try:
        # Query contract by address
        query = (
            db.query(models.Contract)
            .filter(models.Contract.blockchain == models.BlockchainEnum(chain.lower()))
            .filter(models.Contract.address == address)
        )
        db_contract = await db.execute(query)
        db_contract = db_contract.scalar_one_or_none()
        
        if not db_contract:
            raise HTTPException(status_code=404, detail=f"Contract {address} not found")
        
        # Query vulnerabilities for this contract
        query = (
            db.query(models.Vulnerability)
            .filter(models.Vulnerability.contract_id == db_contract.id)
            .order_by(models.Vulnerability.severity)
        )
        result = await db.execute(query)
        vulnerabilities = result.scalars().all()
        
        # Format response
        response = {
            "chain": chain,
            "address": address,
            "contract_name": db_contract.name,
            "is_verified": db_contract.is_verified,
            "vulnerabilities": [],
            "counts": {
                "total": len(vulnerabilities),
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            }
        }
        
        # Process vulnerabilities
        for vuln in vulnerabilities:
            vuln_data = {
                "id": vuln.id,
                "title": vuln.title,
                "description": vuln.description,
                "severity": vuln.severity.value,
                "type": vuln.type,
                "line_number": vuln.line_number,
                "function_name": vuln.function_name,
                "detector": vuln.detector,
                "confidence": vuln.confidence,
                "is_fixed": vuln.is_fixed,
                "created_at": vuln.created_at.timestamp(),
                "updated_at": vuln.updated_at.timestamp()
            }
            
            response["vulnerabilities"].append(vuln_data)
            
            # Update counts
            response["counts"]["total"] += 1
            response["counts"][vuln.severity.value] += 1
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vulnerabilities: {str(e)}")


@router.post("/verify/transaction")
async def verify_zk_transaction(
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    tx_hash: str = Body(..., description="Transaction hash to verify"),
    use_zkevm: bool = Query(False, description="Use zkEVM verification")
):
    """
    Verify a transaction using zero-knowledge proofs.
    
    Verifies the execution and validity of a transaction using ZK proofs.
    For Ethereum, can use zkEVM verification techniques.
    """
    try:
        # Create ZK verifier
        verifier = ZkVerifier()
        
        # Verify transaction
        if use_zkevm and chain.lower() == "ethereum":
            result = await verifier.verify_zkevm_transaction(tx_hash)
        else:
            result = await verifier.verify_transaction(chain, tx_hash)
        
        # Check for verification success
        if not result.get("success", False):
            raise HTTPException(
                status_code=400, 
                detail=result.get("error", "Unknown error during verification")
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error verifying transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying transaction: {str(e)}")


@router.post("/mev/protect")
async def protect_from_mev(
    chain: str = Query("ethereum", description="Blockchain (only ethereum supported currently)"),
    tx_data: Dict[str, Any] = Body(..., description="Transaction data to protect"),
    protection_strategy: str = Query("flashbots", description="Protection strategy to use")
):
    """
    Protect a transaction from MEV attacks.
    
    Applies MEV protection strategies like Flashbots bundles to transactions to avoid
    frontrunning, sandwich attacks, etc.
    """
    if chain.lower() != "ethereum":
        raise HTTPException(status_code=400, detail="MEV protection currently only supports Ethereum")
    
    try:
        from llamachain.security.mev import MevProtection
        
        # Create MEV protection
        mev_protection = MevProtection()
        
        # Apply protection strategy
        result = await mev_protection.protect_transaction(tx_data, strategy=protection_strategy)
        
        return result
        
    except Exception as e:
        logger.error(f"Error protecting transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error protecting transaction: {str(e)}")


@router.get("/security-matrix")
async def get_security_matrix():
    """
    Get the security matrix scores for various blockchain networks and protocols.
    
    Returns a security scoring matrix for different blockchain networks,
    smart contract platforms, and DeFi protocols based on historical vulnerability data.
    """
    # This is a static example - in a real implementation, this would be computed dynamically
    return {
        "blockchains": {
            "ethereum": {
                "score": 87,
                "attack_surface": "medium",
                "historical_vulnerabilities": 14,
                "critical_vulnerabilities": 2
            },
            "solana": {
                "score": 82,
                "attack_surface": "medium",
                "historical_vulnerabilities": 8,
                "critical_vulnerabilities": 1
            },
            "polygon": {
                "score": 84,
                "attack_surface": "medium",
                "historical_vulnerabilities": 6,
                "critical_vulnerabilities": 1
            },
            "arbitrum": {
                "score": 83,
                "attack_surface": "medium",
                "historical_vulnerabilities": 5,
                "critical_vulnerabilities": 0
            }
        },
        "smart_contract_platforms": {
            "ethereum_evm": {
                "score": 85,
                "attack_surface": "high",
                "historical_vulnerabilities": 25,
                "critical_vulnerabilities": 5
            },
            "solana_programs": {
                "score": 81,
                "attack_surface": "medium",
                "historical_vulnerabilities": 12,
                "critical_vulnerabilities": 2
            }
        },
        "defi_protocols": {
            "decentralized_exchanges": {
                "score": 78,
                "attack_surface": "high",
                "historical_vulnerabilities": 32,
                "critical_vulnerabilities": 8
            },
            "lending_platforms": {
                "score": 76,
                "attack_surface": "high",
                "historical_vulnerabilities": 28,
                "critical_vulnerabilities": 6
            },
            "yield_aggregators": {
                "score": 74,
                "attack_surface": "high",
                "historical_vulnerabilities": 30,
                "critical_vulnerabilities": 7
            }
        }
    } 