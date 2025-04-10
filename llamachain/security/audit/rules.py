"""
Rules for security audits.

This module provides rules for detecting security vulnerabilities in smart contracts.
"""

import re
from enum import Enum
from typing import Dict, List, Any, Optional, Pattern, Set, Callable


class Severity(str, Enum):
    """Severity levels for vulnerabilities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class Rule:
    """Rule for detecting a security vulnerability."""
    
    def __init__(
        self,
        vulnerability_type: str,
        title: str,
        description: str,
        severity: Severity,
        patterns: List[str],
        recommendation: Optional[str] = None,
        custom_matcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None
    ):
        """
        Initialize the rule.
        
        Args:
            vulnerability_type: Type of vulnerability
            title: Title of the rule
            description: Description of the vulnerability
            severity: Severity level
            patterns: List of regex patterns to match
            recommendation: Optional recommendation for fixing the vulnerability
            custom_matcher: Optional custom matcher function
        """
        self.vulnerability_type = vulnerability_type
        self.title = title
        self.description = description
        self.severity = severity
        self.patterns = patterns
        self.recommendation = recommendation
        self.custom_matcher = custom_matcher
        
        # Compile regex patterns
        self.compiled_patterns: List[Pattern] = []
        for pattern in patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.MULTILINE | re.DOTALL))
            except re.error:
                # Skip invalid patterns
                pass
    
    def apply(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Apply the rule to source code.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            List of matches, each with position information
        """
        # Use custom matcher if provided
        if self.custom_matcher:
            return self.custom_matcher(source_code)
        
        # Use regex patterns
        matches = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            for match in pattern.finditer(source_code):
                # Calculate line number
                line_number = source_code[:match.start()].count('\n') + 1
                
                # Extract matched text
                matched_text = match.group(0)
                
                # Create match info
                match_info = {
                    "pattern_index": i,
                    "matched_text": matched_text,
                    "position": {
                        "start": match.start(),
                        "end": match.end(),
                        "line": line_number,
                    }
                }
                
                matches.append(match_info)
        
        return matches


class RuleSet:
    """Set of rules for security audits."""
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize the ruleset.
        
        Args:
            rules: Optional list of rules
        """
        self.rules = rules or self._default_rules()
    
    def _default_rules(self) -> List[Rule]:
        """
        Create default rules.
        
        Returns:
            List of default rules
        """
        return [
            # Reentrancy
            Rule(
                vulnerability_type="reentrancy",
                title="Reentrancy Vulnerability",
                description="The contract may be vulnerable to reentrancy attacks, where an external call is made before state changes.",
                severity=Severity.HIGH,
                patterns=[
                    r"(\.\s*transfer\s*\(|\.\s*send\s*\(|\.call\s*\{.*\}\s*\(.*\))(.*)(balance|mapping|state)",
                    r"(balance|mapping|state)(.*)(\.transfer\s*\(|\.send\s*\(|\.call\s*\{.*\}\s*\(.*\))"
                ],
                recommendation="Implement the checks-effects-interactions pattern: perform all state changes before making external calls. Consider using a reentrancy guard."
            ),
            
            # Integer overflow/underflow
            Rule(
                vulnerability_type="integer_overflow",
                title="Integer Overflow/Underflow",
                description="The contract may be vulnerable to integer overflow or underflow.",
                severity=Severity.HIGH,
                patterns=[
                    r"(\+\+|\+=|=\s*\+)(.*)(uint(?!.*SafeMath))",
                    r"(--|-=|=\s*-)(.*)(uint(?!.*SafeMath))"
                ],
                recommendation="Use SafeMath library for arithmetic operations or use Solidity 0.8.0+ which includes built-in overflow checking."
            ),
            
            # Unchecked external call
            Rule(
                vulnerability_type="unchecked_return",
                title="Unchecked External Call",
                description="The contract makes an external call without checking the return value.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"\.transfer\s*\(",
                    r"\.send\s*\((?!.*require)",
                    r"\.call\s*\{.*\}\s*\(.*\)(?!.*require)"
                ],
                recommendation="Always check the return value of low-level calls like send() and call(). Consider using transfer() which reverts on failure."
            ),
            
            # Timestamp dependence
            Rule(
                vulnerability_type="timestamp_dependence",
                title="Timestamp Dependence",
                description="The contract relies on block.timestamp, which can be manipulated by miners.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"block\.timestamp",
                    r"now\s"
                ],
                recommendation="Avoid using block.timestamp for critical logic. If you must use it, ensure that the scale of your time-dependent event is much larger than the scale of potential manipulation (minutes rather than seconds)."
            ),
            
            # Access control issues
            Rule(
                vulnerability_type="access_control",
                title="Missing Access Control",
                description="The contract has functions that should be restricted but lack proper access control.",
                severity=Severity.HIGH,
                patterns=[
                    r"function\s+\w+\s*\([^)]*\)\s*public(?!.*onlyOwner)(?!.*require\s*\(\s*msg\.sender)",
                    r"function\s+\w+\s*\([^)]*\)\s*external(?!.*onlyOwner)(?!.*require\s*\(\s*msg\.sender)"
                ],
                recommendation="Implement proper access control using modifiers like onlyOwner or role-based access control."
            ),
            
            # Use of tx.origin
            Rule(
                vulnerability_type="tx_origin",
                title="Use of tx.origin",
                description="The contract uses tx.origin for authorization, which is vulnerable to phishing attacks.",
                severity=Severity.HIGH,
                patterns=[
                    r"tx\.origin"
                ],
                recommendation="Use msg.sender instead of tx.origin for authorization."
            ),
            
            # Delegatecall to untrusted contract
            Rule(
                vulnerability_type="delegatecall",
                title="Delegatecall to Untrusted Contract",
                description="The contract uses delegatecall with user-supplied input, which can be dangerous.",
                severity=Severity.CRITICAL,
                patterns=[
                    r"\.delegatecall\s*\("
                ],
                recommendation="Avoid using delegatecall with user-supplied input. If necessary, implement strict validation and whitelisting."
            ),
            
            # Uninitialized storage pointer
            Rule(
                vulnerability_type="uninitialized_storage",
                title="Uninitialized Storage Pointer",
                description="The contract has uninitialized storage variables, which can lead to unexpected behavior.",
                severity=Severity.HIGH,
                patterns=[
                    r"(struct\s+\w+\s*\{[^}]*\})(.*)(function\s+\w+\s*\([^)]*\)\s*[^{]*\{[^}]*\w+\s+\w+;)"
                ],
                recommendation="Always initialize storage variables to avoid unexpected behavior."
            ),
            
            # Self-destruct
            Rule(
                vulnerability_type="self_destruct",
                title="Unprotected Self-Destruct",
                description="The contract has an unprotected self-destruct function, which can be called by anyone.",
                severity=Severity.CRITICAL,
                patterns=[
                    r"selfdestruct\s*\((?!.*require\s*\(\s*msg\.sender)",
                    r"suicide\s*\((?!.*require\s*\(\s*msg\.sender)"
                ],
                recommendation="Protect self-destruct functionality with proper access control."
            ),
            
            # Locked ether
            Rule(
                vulnerability_type="locked_ether",
                title="Locked Ether",
                description="The contract can receive ether but has no way to withdraw it.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"(receive\s*\(\)\s*external\s+payable|fallback\s*\(\)\s*external\s+payable|function\s+\w+\s*\([^)]*\)\s*[^{]*payable)(?!.*function\s+\w+\s*\([^)]*\)\s*[^{]*\{\s*[^}]*\.\s*transfer|\.\s*send|\.\s*call)"
                ],
                recommendation="Implement a withdrawal function with proper access control to allow ether to be withdrawn."
            ),
            
            # Outdated compiler version
            Rule(
                vulnerability_type="outdated_compiler",
                title="Outdated Compiler Version",
                description="The contract uses an outdated compiler version, which may have known vulnerabilities.",
                severity=Severity.LOW,
                patterns=[
                    r"pragma\s+solidity\s+(?:0\.[1-7]\.|^0\.8\.0)"
                ],
                recommendation="Use a recent compiler version (0.8.0 or later) to benefit from built-in overflow checking and other security improvements."
            ),
            
            # Floating pragma
            Rule(
                vulnerability_type="floating_pragma",
                title="Floating Pragma",
                description="The contract uses a floating pragma, which may lead to inconsistent behavior across different compiler versions.",
                severity=Severity.LOW,
                patterns=[
                    r"pragma\s+solidity\s+[\^~>]"
                ],
                recommendation="Use a fixed compiler version to ensure consistent behavior."
            ),
            
            # Use of deprecated functions
            Rule(
                vulnerability_type="deprecated_functions",
                title="Use of Deprecated Functions",
                description="The contract uses deprecated functions, which may be removed in future compiler versions.",
                severity=Severity.LOW,
                patterns=[
                    r"suicide\s*\(",
                    r"block\.blockhash",
                    r"sha3\s*\("
                ],
                recommendation="Replace deprecated functions with their modern equivalents: use selfdestruct() instead of suicide(), blockhash() instead of block.blockhash(), and keccak256() instead of sha3()."
            ),
            
            # Unchecked math
            Rule(
                vulnerability_type="unchecked_math",
                title="Unchecked Math Operations",
                description="The contract performs math operations without checking for overflow/underflow.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"(\+\+|\+=|=\s*\+|-=|=\s*-|\*=|=\s*\*|/=|=\s*/|%=|=\s*%)(?!.*SafeMath)"
                ],
                recommendation="Use SafeMath library for arithmetic operations or use Solidity 0.8.0+ which includes built-in overflow checking."
            ),
            
            # Gas limit issues
            Rule(
                vulnerability_type="gas_limit",
                title="Gas Limit Issues",
                description="The contract may have functions that could exceed the block gas limit.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"for\s*\([^;]*;\s*[^;]*;\s*[^)]*\)\s*\{[^}]*\}"
                ],
                recommendation="Avoid unbounded loops and operations on large arrays. Consider implementing pagination or batching for large operations."
            ),
            
            # Private data exposure
            Rule(
                vulnerability_type="private_data",
                title="Private Data Exposure",
                description="The contract stores sensitive data as private, but this data is still visible on the blockchain.",
                severity=Severity.LOW,
                patterns=[
                    r"private\s+\w+\s+password",
                    r"private\s+\w+\s+secret",
                    r"private\s+\w+\s+key"
                ],
                recommendation="Do not store sensitive data on the blockchain, even as private variables. All data on the blockchain is publicly visible."
            ),
            
            # Weak randomness
            Rule(
                vulnerability_type="weak_randomness",
                title="Weak Randomness",
                description="The contract uses weak sources of randomness, which can be predicted or manipulated.",
                severity=Severity.HIGH,
                patterns=[
                    r"block\.timestamp\s*[+*/%^&|]",
                    r"blockhash\s*\(",
                    r"block\.difficulty",
                    r"block\.coinbase",
                    r"block\.number\s*[+*/%^&|]"
                ],
                recommendation="Do not rely on blockchain properties for randomness. Consider using a verifiable random function (VRF) or an oracle for secure randomness."
            ),
            
            # Front-running
            Rule(
                vulnerability_type="front_running",
                title="Front-Running Vulnerability",
                description="The contract may be vulnerable to front-running attacks, where transactions can be observed and exploited.",
                severity=Severity.MEDIUM,
                patterns=[
                    r"function\s+\w+\s*\([^)]*\)\s*[^{]*\{\s*[^}]*require\s*\([^)]*\)\s*;\s*[^}]*\.\s*transfer"
                ],
                recommendation="Implement mechanisms like commit-reveal schemes or use a minimum/maximum value to prevent front-running attacks."
            ),
        ]
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the ruleset.
        
        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
    
    def remove_rule(self, vulnerability_type: str) -> None:
        """
        Remove rules of a specific vulnerability type.
        
        Args:
            vulnerability_type: Type of vulnerability to remove
        """
        self.rules = [rule for rule in self.rules if rule.vulnerability_type != vulnerability_type]
    
    def get_rules_by_severity(self, severity: Severity) -> List[Rule]:
        """
        Get rules by severity.
        
        Args:
            severity: Severity level
            
        Returns:
            List of rules with the specified severity
        """
        return [rule for rule in self.rules if rule.severity == severity] 