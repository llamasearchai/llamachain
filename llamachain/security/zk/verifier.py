"""
Zero-knowledge proof verifier.

This module provides functionality for verifying zero-knowledge proofs.
"""

import json
import logging
import subprocess
import tempfile
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING, Tuple, cast

from llamachain.core.exceptions import SecurityError
from llamachain.log import get_logger

# Handle py_ecc imports for type checking - making the imports more specific
if TYPE_CHECKING:
    # Import specific names from py_ecc.bn128 to help Pylance resolve them
    from py_ecc.bn128 import (
        G1, G2, 
        add, curve_order, multiply, neg, pairing
    )

# Get logger
logger = get_logger("llamachain.security.zk.verifier")


class ProofSystem(str, Enum):
    """Supported zero-knowledge proof systems."""
    GROTH16 = "groth16"
    PLONK = "plonk"
    STARK = "stark"
    BULLETPROOFS = "bulletproofs"
    SNARKJS = "snarkjs"


class ProofStatus(str, Enum):
    """Status of a proof verification."""
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"
    UNKNOWN = "unknown"


class ZKVerifier:
    """Verifier for zero-knowledge proofs."""
    
    def __init__(self):
        """Initialize the ZK verifier."""
        # Check if py_ecc is available for elliptic curve operations
        try:
            # Use explicit relative imports to help Pylance locate the module
            import py_ecc.bn128
            self._bn128 = py_ecc.bn128
            self.py_ecc_available = True
            logger.info("py_ecc.bn128 successfully imported")
        except ImportError as e:
            self._bn128 = None
            self.py_ecc_available = False
            logger.warning(f"py_ecc library not available, some verification methods will be limited: {e}")
        
        # Check if snarkjs is available for snarkjs proofs
        try:
            result = subprocess.run(
                ["node", "-e", "console.log(require.resolve('snarkjs'))"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            self.snarkjs_available = result.returncode == 0
            if self.snarkjs_available:
                logger.info("snarkjs successfully detected")
            else:
                logger.warning("snarkjs not available, snarkjs verification will be limited")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.snarkjs_available = False
            logger.warning(f"snarkjs not available, snarkjs verification will be limited: {e}")
        
        logger.info(f"ZK verifier initialized (py_ecc: {self.py_ecc_available}, snarkjs: {self.snarkjs_available})")
    
    async def verify_proof(
        self,
        proof_system: ProofSystem,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a zero-knowledge proof.
        
        Args:
            proof_system: The proof system used
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        try:
            if proof_system == ProofSystem.GROTH16:
                return await self._verify_groth16(proof, verification_key, public_inputs)
            elif proof_system == ProofSystem.PLONK:
                return await self._verify_plonk(proof, verification_key, public_inputs)
            elif proof_system == ProofSystem.STARK:
                return await self._verify_stark(proof, verification_key, public_inputs)
            elif proof_system == ProofSystem.BULLETPROOFS:
                return await self._verify_bulletproofs(proof, verification_key, public_inputs)
            elif proof_system == ProofSystem.SNARKJS:
                return await self._verify_snarkjs(proof, verification_key, public_inputs)
            else:
                logger.error(f"Unsupported proof system: {proof_system}")
                return ProofStatus.UNKNOWN
        except Exception as e:
            logger.error(f"Error verifying proof: {e}")
            return ProofStatus.ERROR
    
    async def _verify_groth16(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a Groth16 proof.
        
        Args:
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        if not self.py_ecc_available or not self._bn128:
            logger.warning("py_ecc library not available, cannot verify Groth16 proof")
            return ProofStatus.UNKNOWN
        
        try:
            # Use the module we stored during initialization
            bn128 = self._bn128
            
            # Extract proof components
            a = proof.get("a", [])
            b = proof.get("b", [])
            c = proof.get("c", [])
            
            # Extract verification key components
            alpha = verification_key.get("alpha", [])
            beta = verification_key.get("beta", [])
            gamma = verification_key.get("gamma", [])
            delta = verification_key.get("delta", [])
            ic = verification_key.get("ic", [])
            
            # Validate inputs
            if not a or not b or not c:
                logger.error("Invalid proof format")
                return ProofStatus.INVALID
            
            if not alpha or not beta or not gamma or not delta or not ic:
                logger.error("Invalid verification key format")
                return ProofStatus.INVALID
            
            # Convert proof components to points
            a_point = bn128.G1(a[0], a[1])
            b_point = bn128.G2(b[0][0], b[0][1], b[1][0], b[1][1])
            c_point = bn128.G1(c[0], c[1])
            
            # Convert verification key components to points
            alpha_point = bn128.G1(alpha[0], alpha[1])
            beta_point = bn128.G2(beta[0][0], beta[0][1], beta[1][0], beta[1][1])
            gamma_point = bn128.G2(gamma[0][0], gamma[0][1], gamma[1][0], gamma[1][1])
            delta_point = bn128.G2(delta[0][0], delta[0][1], delta[1][0], delta[1][1])
            
            # Compute linear combination of public inputs
            vk_x = bn128.G1(ic[0][0], ic[0][1])
            if public_inputs:
                for i, input_value in enumerate(public_inputs):
                    if i + 1 < len(ic):
                        input_point = bn128.G1(ic[i + 1][0], ic[i + 1][1])
                        vk_x = bn128.add(vk_x, bn128.multiply(input_point, int(input_value)))
            
            # Verify the pairing equation
            # e(A, B) * e(-vk_x, gamma) * e(-C, delta) == e(alpha, beta)
            neg_vk_x = bn128.neg(vk_x)
            neg_c = bn128.neg(c_point)
            
            left_pairing = bn128.pairing(a_point, b_point) * bn128.pairing(neg_vk_x, gamma_point) * bn128.pairing(neg_c, delta_point)
            right_pairing = bn128.pairing(alpha_point, beta_point)
            
            if left_pairing == right_pairing:
                return ProofStatus.VALID
            else:
                return ProofStatus.INVALID
        
        except Exception as e:
            logger.error(f"Error verifying Groth16 proof: {e}")
            return ProofStatus.ERROR
    
    async def _verify_plonk(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a PLONK proof.
        
        Args:
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        # PLONK verification is complex and requires a specialized library
        # This is a placeholder for future implementation
        logger.warning("PLONK verification not fully implemented")
        return ProofStatus.UNKNOWN
    
    async def _verify_stark(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a STARK proof.
        
        Args:
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        # STARK verification is complex and requires a specialized library
        # This is a placeholder for future implementation
        logger.warning("STARK verification not fully implemented")
        return ProofStatus.UNKNOWN
    
    async def _verify_bulletproofs(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a Bulletproofs proof.
        
        Args:
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        # Bulletproofs verification is complex and requires a specialized library
        # This is a placeholder for future implementation
        logger.warning("Bulletproofs verification not fully implemented")
        return ProofStatus.UNKNOWN
    
    async def _verify_snarkjs(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any],
        public_inputs: Optional[List[str]] = None
    ) -> ProofStatus:
        """
        Verify a snarkjs proof.
        
        Args:
            proof: The proof to verify
            verification_key: The verification key
            public_inputs: Optional public inputs
            
        Returns:
            Status of the verification
        """
        if not self.snarkjs_available:
            logger.warning("snarkjs not available, cannot verify snarkjs proof")
            return ProofStatus.UNKNOWN
        
        try:
            # Create temporary files for proof, verification key, and public inputs
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as proof_file:
                json.dump(proof, proof_file)
                proof_file_path = proof_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as vk_file:
                json.dump(verification_key, vk_file)
                vk_file_path = vk_file.name
            
            public_inputs_file_path = None
            if public_inputs:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as public_inputs_file:
                    json.dump(public_inputs, public_inputs_file)
                    public_inputs_file_path = public_inputs_file.name
            
            # Run snarkjs verification
            cmd = [
                "node", "-e",
                f"const snarkjs = require('snarkjs'); snarkjs.groth16.verify(require('{vk_file_path}'), {json.dumps(public_inputs) if public_inputs else '[]'}, require('{proof_file_path}')).then(valid => console.log(valid));"
            ]
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            # Clean up temporary files
            os.unlink(proof_file_path)
            os.unlink(vk_file_path)
            if public_inputs_file_path:
                os.unlink(public_inputs_file_path)
            
            # Parse output
            if process.returncode == 0:
                output = process.stdout.decode("utf-8").strip()
                if output == "true":
                    return ProofStatus.VALID
                elif output == "false":
                    return ProofStatus.INVALID
            
            logger.error(f"Error verifying snarkjs proof: {process.stderr.decode('utf-8')}")
            return ProofStatus.ERROR
        
        except Exception as e:
            logger.error(f"Error verifying snarkjs proof: {e}")
            return ProofStatus.ERROR
    
    def generate_verification_key(
        self,
        proof_system: ProofSystem,
        circuit_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a verification key for a circuit.
        
        Args:
            proof_system: The proof system to use
            circuit_params: Parameters for the circuit
            
        Returns:
            Verification key
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would generate a verification key based on the circuit
        logger.warning("Verification key generation not fully implemented")
        
        # Return a dummy verification key
        return {
            "type": proof_system.value,
            "params": circuit_params,
            "generated": True,
        } 