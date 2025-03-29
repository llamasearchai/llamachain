"""
Contract analyzer for security audits.

This module provides functionality for analyzing smart contracts for security vulnerabilities.
"""

import os
import re
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from llamachain.core.constants import VULNERABILITY_TYPES, AUDIT_SEVERITY_LEVELS
from llamachain.core.exceptions import SecurityError
from llamachain.log import get_logger
from llamachain.security.audit.rules import Rule, RuleSet, Severity

# Get logger
logger = get_logger("llamachain.security.audit.analyzer")


@dataclass
class VulnerabilityInfo:
    """Information about a detected vulnerability."""
    
    type: str
    severity: str
    title: str
    description: str
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
        }


@dataclass
class AuditResult:
    """Result of a contract audit."""
    
    contract_address: str
    chain_id: str
    vulnerabilities: List[VulnerabilityInfo] = field(default_factory=list)
    audit_score: int = 100
    audit_type: str = "static_analysis"
    report: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_address": self.contract_address,
            "chain_id": self.chain_id,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "vulnerabilities_count": len(self.vulnerabilities),
            "audit_score": self.audit_score,
            "audit_type": self.audit_type,
            "report": self.report,
        }
    
    def add_vulnerability(self, vulnerability: VulnerabilityInfo) -> None:
        """Add a vulnerability to the audit result."""
        self.vulnerabilities.append(vulnerability)
        
        # Update audit score based on severity
        severity_scores = {
            "critical": 30,
            "high": 20,
            "medium": 10,
            "low": 5,
            "informational": 0,
        }
        
        score_reduction = severity_scores.get(vulnerability.severity.lower(), 0)
        self.audit_score = max(0, self.audit_score - score_reduction)


class ContractAuditor:
    """Auditor for smart contracts."""
    
    def __init__(self, ruleset: Optional[RuleSet] = None):
        """
        Initialize the contract auditor.
        
        Args:
            ruleset: Optional ruleset to use for auditing
        """
        self.ruleset = ruleset or RuleSet()
        
        # Check if external tools are available
        self.mythril_available = self._check_tool_available("myth")
        self.slither_available = self._check_tool_available("slither")
        
        logger.info(f"Contract auditor initialized (Mythril: {self.mythril_available}, Slither: {self.slither_available})")
    
    def _check_tool_available(self, tool_name: str) -> bool:
        """
        Check if an external tool is available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if the tool is available, False otherwise
        """
        try:
            subprocess.run(
                [tool_name, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def audit_contract(
        self,
        contract_address: str,
        chain_id: str,
        source_code: Optional[str] = None,
        bytecode: Optional[str] = None,
        use_external_tools: bool = True
    ) -> AuditResult:
        """
        Audit a smart contract.
        
        Args:
            contract_address: Address of the contract
            chain_id: Blockchain ID
            source_code: Optional source code of the contract
            bytecode: Optional bytecode of the contract
            use_external_tools: Whether to use external tools like Mythril and Slither
            
        Returns:
            Audit result
        """
        if not source_code and not bytecode:
            raise SecurityError("Either source code or bytecode must be provided")
        
        # Initialize audit result
        result = AuditResult(
            contract_address=contract_address,
            chain_id=chain_id
        )
        
        # Perform static analysis on source code
        if source_code:
            self._analyze_source_code(source_code, result)
            
            # Use external tools if available and requested
            if use_external_tools:
                if self.slither_available:
                    self._analyze_with_slither(source_code, result)
                
                if self.mythril_available and bytecode:
                    self._analyze_with_mythril(bytecode, result)
        
        # Perform bytecode analysis if no source code is available
        elif bytecode and use_external_tools and self.mythril_available:
            self._analyze_with_mythril(bytecode, result)
        
        # Generate report
        result.report = self._generate_report(result)
        
        return result
    
    def _analyze_source_code(self, source_code: str, result: AuditResult) -> None:
        """
        Analyze contract source code using internal rules.
        
        Args:
            source_code: Source code of the contract
            result: Audit result to update
        """
        # Apply each rule to the source code
        for rule in self.ruleset.rules:
            matches = rule.apply(source_code)
            
            for match in matches:
                # Extract line number and code snippet
                line_number = None
                code_snippet = None
                function_name = None
                
                if match.get("position"):
                    line_number = match["position"].get("line")
                    
                    # Extract code snippet around the line
                    if line_number:
                        lines = source_code.splitlines()
                        start_line = max(0, line_number - 3)
                        end_line = min(len(lines), line_number + 3)
                        code_snippet = "\n".join(lines[start_line:end_line])
                
                # Extract function name if available
                if match.get("function_name"):
                    function_name = match["function_name"]
                elif line_number:
                    # Try to extract function name from the code
                    lines = source_code.splitlines()
                    for i in range(max(0, line_number - 5), min(len(lines), line_number)):
                        function_match = re.search(r"function\s+(\w+)", lines[i])
                        if function_match:
                            function_name = function_match.group(1)
                            break
                
                # Create vulnerability info
                vulnerability = VulnerabilityInfo(
                    type=rule.vulnerability_type,
                    severity=rule.severity.value,
                    title=rule.title,
                    description=rule.description,
                    line_number=line_number,
                    function_name=function_name,
                    code_snippet=code_snippet,
                    recommendation=rule.recommendation
                )
                
                # Add to result
                result.add_vulnerability(vulnerability)
    
    def _analyze_with_slither(self, source_code: str, result: AuditResult) -> None:
        """
        Analyze contract source code using Slither.
        
        Args:
            source_code: Source code of the contract
            result: Audit result to update
        """
        try:
            # Create temporary file with source code
            with tempfile.NamedTemporaryFile(suffix=".sol", delete=False) as temp_file:
                temp_file.write(source_code.encode("utf-8"))
                temp_file_path = temp_file.name
            
            # Run Slither
            process = subprocess.run(
                ["slither", temp_file_path, "--json", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Parse Slither output
            if process.returncode == 0 and process.stdout:
                try:
                    slither_output = json.loads(process.stdout.decode("utf-8"))
                    
                    # Extract vulnerabilities
                    for detector in slither_output.get("detectors", []):
                        # Map Slither impact to severity
                        impact = detector.get("impact", "").lower()
                        severity = "medium"  # Default
                        
                        if impact == "high":
                            severity = "high"
                        elif impact == "medium":
                            severity = "medium"
                        elif impact == "low":
                            severity = "low"
                        elif impact == "informational":
                            severity = "informational"
                        
                        # Extract line number and code snippet
                        line_number = None
                        code_snippet = None
                        function_name = None
                        
                        if detector.get("elements"):
                            for element in detector["elements"]:
                                if element.get("source_mapping", {}).get("lines"):
                                    lines = element["source_mapping"]["lines"]
                                    if lines:
                                        line_number = lines[0]
                                
                                if element.get("name"):
                                    function_name = element["name"]
                        
                        # Create vulnerability info
                        vulnerability = VulnerabilityInfo(
                            type=detector.get("check", "unknown"),
                            severity=severity,
                            title=detector.get("check", "Unknown vulnerability"),
                            description=detector.get("description", ""),
                            line_number=line_number,
                            function_name=function_name,
                            code_snippet=code_snippet,
                            recommendation=detector.get("recommendation", "")
                        )
                        
                        # Add to result
                        result.add_vulnerability(vulnerability)
                
                except json.JSONDecodeError:
                    logger.error("Failed to parse Slither output")
        
        except Exception as e:
            logger.error(f"Error running Slither: {e}")
    
    def _analyze_with_mythril(self, bytecode: str, result: AuditResult) -> None:
        """
        Analyze contract bytecode using Mythril.
        
        Args:
            bytecode: Bytecode of the contract
            result: Audit result to update
        """
        try:
            # Run Mythril
            process = subprocess.run(
                ["myth", "analyze", "-c", bytecode, "-o", "json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            # Parse Mythril output
            if process.returncode == 0 and process.stdout:
                try:
                    mythril_output = json.loads(process.stdout.decode("utf-8"))
                    
                    # Extract vulnerabilities
                    for issue in mythril_output.get("issues", []):
                        # Map Mythril severity to our severity
                        severity = "medium"  # Default
                        
                        if issue.get("severity") == "High":
                            severity = "high"
                        elif issue.get("severity") == "Medium":
                            severity = "medium"
                        elif issue.get("severity") == "Low":
                            severity = "low"
                        
                        # Create vulnerability info
                        vulnerability = VulnerabilityInfo(
                            type=issue.get("swc-id", "unknown"),
                            severity=severity,
                            title=issue.get("title", "Unknown vulnerability"),
                            description=issue.get("description", ""),
                            line_number=None,  # Mythril doesn't provide line numbers for bytecode analysis
                            function_name=None,
                            code_snippet=None,
                            recommendation=issue.get("recommendation", "")
                        )
                        
                        # Add to result
                        result.add_vulnerability(vulnerability)
                
                except json.JSONDecodeError:
                    logger.error("Failed to parse Mythril output")
        
        except Exception as e:
            logger.error(f"Error running Mythril: {e}")
    
    def _generate_report(self, result: AuditResult) -> Dict[str, Any]:
        """
        Generate a detailed audit report.
        
        Args:
            result: Audit result
            
        Returns:
            Report as a dictionary
        """
        # Count vulnerabilities by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "informational": 0,
        }
        
        for vuln in result.vulnerabilities:
            severity = vuln.severity.lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Generate report
        report = {
            "summary": {
                "contract_address": result.contract_address,
                "chain_id": result.chain_id,
                "audit_score": result.audit_score,
                "vulnerabilities_count": len(result.vulnerabilities),
                "severity_counts": severity_counts,
            },
            "vulnerabilities_by_severity": {},
            "recommendations": [],
        }
        
        # Group vulnerabilities by severity
        for severity in AUDIT_SEVERITY_LEVELS:
            report["vulnerabilities_by_severity"][severity] = []
        
        for vuln in result.vulnerabilities:
            severity = vuln.severity.lower()
            if severity in report["vulnerabilities_by_severity"]:
                report["vulnerabilities_by_severity"][severity].append(vuln.to_dict())
            
            # Add recommendation if available
            if vuln.recommendation:
                report["recommendations"].append({
                    "title": vuln.title,
                    "recommendation": vuln.recommendation,
                })
        
        return report 