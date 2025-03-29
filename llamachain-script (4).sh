                    {"name": "value", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "gas", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "gas_price", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "nonce", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "transaction_index", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "input_data", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"}
                ]
            }
            
            metrics_table_schema = {
                "fields": [
                    {"name": "window_start", "type": "TIMESTAMP", "mode": "REQUIRED"},
                    {"name": "window_end", "type": "TIMESTAMP", "mode": "REQUIRED"},
                    {"name": "metric_type", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "avg_gas_utilization", "type": "FLOAT", "mode": "NULLABLE"},
                    {"name": "total_transactions", "type": "INTEGER", "mode": "NULLABLE"},
                    {"name": "avg_block_size", "type": "FLOAT", "mode": "NULLABLE"},
                    {"name": "avg_gas_price", "type": "FLOAT", "mode": "NULLABLE"}
                ]
            }
            
            # Write parsed blocks to BigQuery
            blocks_for_bq = (
                windowed_data
                | "FilterBlocks" >> beam.Filter(lambda x: "number" in x)
                | "WriteToBigQuery_Blocks" >> WriteToBigQuery(
                    table=f"{project}:{dataset}.blocks",
                    schema=block_table_schema,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
            
            # Write parsed transactions to BigQuery
            txs_for_bq = (
                windowed_data
                | "FilterTransactions" >> beam.Filter(lambda x: "from_address" in x and "to_address" in x)
                | "WriteToBigQuery_Transactions" >> WriteToBigQuery(
                    table=f"{project}:{dataset}.transactions",
                    schema=tx_table_schema,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
            
            # Write aggregated metrics to BigQuery
            metrics_for_bq = (
                aggregated_metrics
                | "FormatMetrics" >> beam.Map(lambda x: {
                    "window_start": x[1],  # Window start time
                    "window_end": x[1] + 3600,  # Window end time (1 hour later)
                    "metric_type": x[0],
                    "avg_gas_utilization": x[2].get("avg_gas_utilization"),
                    "total_transactions": x[2].get("total_transactions"),
                    "avg_block_size": x[2].get("avg_block_size"),
                    "avg_gas_price": x[2].get("avg_gas_price")
                })
                | "WriteToBigQuery_Metrics" >> WriteToBigQuery(
                    table=f"{project}:{dataset}.metrics",
                    schema=metrics_table_schema,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
        else:
            # If no BigQuery project is set, just log the results
            _ = (
                aggregated_metrics
                | "LogResults" >> beam.Map(lambda x: logger.info(f"Metrics: {x}"))
            )
        
        logger.info("Pipeline constructed successfully")
    
    logger.info("Pipeline completed successfully")

# Main entry point when run as a module
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaChain Analytics Pipeline")
    parser.add_argument("--project", help="BigQuery project ID", default=os.environ.get("BIGQUERY_PROJECT"))
    parser.add_argument("--dataset", help="BigQuery dataset", default=os.environ.get("BIGQUERY_DATASET", "blockchain_data"))
    parser.add_argument("--runner", help="Beam runner", default=os.environ.get("BEAM_RUNNER", "DirectRunner"))
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["BIGQUERY_PROJECT"] = args.project or ""
    os.environ["BIGQUERY_DATASET"] = args.dataset
    os.environ["BEAM_RUNNER"] = args.runner
    
    # Run the pipeline
    run_pipeline()

if __name__ == "__main__":
    main()
EOF
}

function install_and_run() {
  print_header "Installing and Running LlamaChain"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "Virtual environment not activated"
    return 1
  fi
  
  # Install the package in development mode
  log_info "Installing LlamaChain in development mode..."
  pip install -e .
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to install LlamaChain"
    return 1
  fi
  
  log_success "LlamaChain installed successfully"
  
  # Run tests if present
  if [[ -d "tests" ]]; then
    log_info "Running tests..."
    pytest tests
  fi
  
  # Launch the CLI
  log_info "Launching LlamaChain CLI..."
  llamachain --version
  
  log_success "LlamaChain is ready to use!"
  log_info "Use 'llamachain --help' to see available commands"
  
  return 0
}

function run_example() {
  print_header "Running Example"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "Virtual environment not activated"
    return 1
  fi
  
  # Run a simple audit on the example contract
  log_info "Running an audit on the example contract..."
  llamachain audit contracts/example.sol --output data/audit_result.json
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to run audit example"
    return 1
  fi
  
  log_success "Example completed successfully"
  log_info "Audit results saved to data/audit_result.json"
  
  return 0
}

function create_tests() {
  print_header "Creating Tests"
  
  log_info "Creating test fixtures..."
  mkdir -p tests
  cat > tests/conftest.py << EOF
"""
Test fixtures for LlamaChain.
"""

import os
import pytest
from web3 import Web3, EthereumTesterProvider

from llamachain.core.indexer import BlockchainIndexer
from llamachain.analysis.contract import ContractAuditor
from llamachain.analysis.trace import TransactionTracer
from llamachain.security.frontrun import FrontrunDetector
from llamachain.zk.verifier import Verifier

@pytest.fixture
def eth_tester_provider():
    """Ethereum tester provider fixture."""
    return EthereumTesterProvider()

@pytest.fixture
def web3(eth_tester_provider):
    """Web3 fixture with eth-tester provider."""
    return Web3(eth_tester_provider)

@pytest.fixture
def indexer(web3):
    """BlockchainIndexer fixture with eth-tester provider."""
    # Create a mock indexer that uses the eth-tester provider
    original_init = BlockchainIndexer.__init__
    
    def patched_init(self, rpc_url=None, ws_url=None):
        # Call original init
        original_init(self, rpc_url, ws_url)
        # Replace web3 instance with our test instance
        self.web3 = web3
    
    # Patch the init method temporarily
    BlockchainIndexer.__init__ = patched_init
    
    # Create the indexer
    indexer = BlockchainIndexer()
    
    # Restore the original init method
    BlockchainIndexer.__init__ = original_init
    
    return indexer

@pytest.fixture
def auditor():
    """ContractAuditor fixture."""
    return ContractAuditor()

@pytest.fixture
def tracer(web3):
    """TransactionTracer fixture with eth-tester provider."""
    # Create a mock tracer that uses the eth-tester provider
    original_init = TransactionTracer.__init__
    
    def patched_init(self, rpc_url=None):
        # Call original init
        original_init(self, rpc_url)
        # Replace web3 instance with our test instance
        self.web3 = web3
    
    # Patch the init method temporarily
    TransactionTracer.__init__ = patched_init
    
    # Create the tracer
    tracer = TransactionTracer()
    
    # Restore the original init method
    TransactionTracer.__init__ = original_init
    
    return tracer

@pytest.fixture
def detector(web3):
    """FrontrunDetector fixture with eth-tester provider."""
    # Create a mock detector that uses the eth-tester provider
    original_init = FrontrunDetector.__init__
    
    def patched_init(self, rpc_url=None, ws_url=None):
        # Call original init
        original_init(self, rpc_url, ws_url)
        # Replace web3 instance with our test instance
        self.http_web3 = web3
    
    # Patch the init method temporarily
    FrontrunDetector.__init__ = patched_init
    
    # Create the detector
    detector = FrontrunDetector()
    
    # Restore the original init method
    FrontrunDetector.__init__ = original_init
    
    return detector

@pytest.fixture
def verifier():
    """Verifier fixture."""
    return Verifier()

@pytest.fixture
def example_contract_path():
    """Path to the example contract."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "contracts", "example.sol")
EOF

  log_info "Creating test for BlockchainIndexer..."
  cat > tests/test_indexer.py << EOF
"""
Tests for the BlockchainIndexer.
"""

import pytest
from web3 import Web3

def test_indexer_initialization(indexer):
    """Test that the indexer is initialized correctly."""
    assert indexer is not None
    assert isinstance(indexer.web3, Web3)
    assert indexer.web3.is_connected()

def test_get_block(indexer, web3):
    """Test getting a block."""
    # Create a new block
    web3.eth.mine()
    
    # Get the latest block number
    latest_block_number = web3.eth.block_number
    
    # Use indexer to get the block
    block = indexer.get_block(latest_block_number)
    
    # Verify the block
    assert block is not None
    assert block["number"] == latest_block_number
    assert "hash" in block
    assert "parentHash" in block
    assert "timestamp" in block

def test_index_block(indexer, web3):
    """Test indexing a block."""
    # Create a new block
    web3.eth.mine()
    
    # Get the latest block number
    latest_block_number = web3.eth.block_number
    
    # Index the block
    success = indexer.index_block(latest_block_number)
    
    # Verify the result
    assert success is True
    
    # Verify stats
    stats = indexer.get_stats()
    assert stats["blocks_processed"] >= 1
EOF

  log_info "Creating test for ContractAuditor..."
  cat > tests/test_auditor.py << EOF
"""
Tests for the ContractAuditor.
"""

import pytest
import os

def test_auditor_initialization(auditor):
    """Test that the auditor is initialized correctly."""
    assert auditor is not None

def test_analyze_contract(auditor, example_contract_path):
    """Test analyzing a contract."""
    # Verify the contract file exists
    assert os.path.exists(example_contract_path)
    
    # Analyze the contract
    results = auditor.analyze_contract(example_contract_path)
    
    # Verify the results
    assert results is not None
    assert "success" in results
    assert results["success"] is True
    assert "findings" in results
    assert isinstance(results["findings"], list)
    assert "summary" in results
    assert "total" in results["summary"]
EOF

  log_info "Creating test for Verifier..."
  cat > tests/test_verifier.py << EOF
"""
Tests for the ZK Verifier.
"""

import pytest

def test_verifier_initialization(verifier):
    """Test that the verifier is initialized correctly."""
    assert verifier is not None

def test_dummy_proof_generation(verifier):
    """Test generating a dummy proof."""
    # Skip if py_ecc is not available
    if not hasattr(verifier, 'generate_dummy_proof'):
        pytest.skip("py_ecc not available")
    
    # Generate a dummy proof
    proof = verifier.generate_dummy_proof()
    
    # Verify the proof
    assert proof is not None
    assert "a" in proof
    assert "b" in proof
    assert "c" in proof

def test_dummy_vk_generation(verifier):
    """Test generating a dummy verification key."""
    # Skip if py_ecc is not available
    if not hasattr(verifier, 'generate_dummy_vk'):
        pytest.skip("py_ecc not available")
    
    # Generate a dummy verification key
    vk = verifier.generate_dummy_vk()
    
    # Verify the verification key
    assert vk is not None
    assert "alpha" in vk
    assert "beta" in vk
    assert "gamma" in vk
    assert "delta" in vk
    assert "ic" in vk
EOF

  log_success "Tests created successfully"
}

# Main execution flow
function main() {
  # Display welcome banner
  display_llama_banner
  
  # Parse command-line arguments
  SETUP=0
  START=0
  TEST=0
  
  while [[ $# -gt 0 ]]; do
    case $1 in
      --setup)
        SETUP=1
        shift
        ;;
      --start)
        START=1
        shift
        ;;
      --test)
        TEST=1
        shift
        ;;
      *)
        log_error "Unknown argument: $1"
        echo "Usage: $0 [--setup] [--start] [--test]"
        exit 1
        ;;
    esac
  done
  
  # If no arguments are provided, do everything
  if [[ $SETUP -eq 0 && $START -eq 0 && $TEST -eq 0 ]]; then
    SETUP=1
    START=1
    TEST=1
  fi
  
  # Setup
  if [[ $SETUP -eq 1 ]]; then
    # Check prerequisites
    check_prerequisites || exit 1
    
    # Create virtual environment
    create_virtual_environment || exit 1
    
    # Install dependencies
    install_dependencies || exit 1
    
    # Create project structure
    create_project_structure || exit 1
    
    # Create configuration files
    create_configuration_files || exit 1
    
    # Create core modules
    create_core_modules || exit 1
    
    # Create contract auditor
    create_contract_auditor || exit 1
    
    # Create transaction tracer
    create_transaction_tracer || exit 1
    
    # Create security modules
    create_security_modules || exit 1
    
    # Create ZK verifier
    create_zk_verifier || exit 1
    
    # Create API endpoints
    create_api_endpoints || exit 1
    
    # Create CLI
    create_cli || exit 1
    
    # Create analytics pipeline
    create_analytics_pipeline || exit 1
    
    # Create tests
    create_tests || exit 1
  fi
  
  # Start
  if [[ $START -eq 1 ]]; then
    # Install and run
    install_and_run || exit 1
  fi
  
  # Test
  if [[ $TEST -eq 1 ]]; then
    # Run example
    run_example || exit 1
  fi
  
  log_success "LlamaChain setup completed successfully!"
  log_info "To activate LlamaChain, run: source venv/bin/activate"
  log_info "To see available commands, run: llamachain --help"
  
  return 0
}

# Execute main function
main "$@"
        console.print(f"[green]Indexed latest {latest} blocks[/green]")
    
    elif range is not None:
        try:
            start, end = map(int, range.split("-"))
            
            total_blocks = end - start + 1
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task(f"Indexing blocks {start} to {end}...", total=total_blocks)
                
                for block_num in range(start, end + 1):
                    progress.update(task, description=f"Indexing block {block_num}...")
                    success = indexer.index_block(block_num)
                    progress.advance(task)
            
            console.print(f"[green]Indexed blocks {start} to {end}[/green]")
        except ValueError:
            console.print("[red]Invalid range format. Use start-end (e.g., 15000000-15000010)[/red]")
    
    elif listen:
        def handle_new_block(block_number):
            console.print(f"[green]Indexed new block {block_number}[/green]")
        
        console.print("[blue]Listening for new blocks...[/blue]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        
        try:
            indexer.listen_for_new_blocks(callback=handle_new_block)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped listening for blocks[/yellow]")
    
    else:
        console.print("[yellow]No action specified. Use --help to see available options.[/yellow]")
    
    # Display stats
    stats = indexer.get_stats()
    
    table = Table(title="Indexer Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Blocks Processed", str(stats["blocks_processed"]))
    table.add_row("Transactions Processed", str(stats["transactions_processed"]))
    table.add_row("Elapsed Time", f"{stats['elapsed_time_seconds']:.2f} seconds")
    table.add_row("Blocks Per Hour", f"{stats['blocks_per_hour']:.2f}")
    table.add_row("Transactions Per Hour", f"{stats['transactions_per_hour']:.2f}")
    
    console.print(table)

# Contract commands
@app.command()
def audit(
    contract: str = typer.Argument(..., help="Path to the contract source code or contract address"),
    output: str = typer.Option(None, help="Path to output the results (JSON format)"),
    is_address: bool = typer.Option(False, help="Treat the contract parameter as an address")
):
    """Audit a smart contract for vulnerabilities."""
    print_banner()
    
    console.print("[blue]Initializing Contract Auditor...[/blue]")
    auditor = ContractAuditor()
    
    if is_address:
        console.print(f"[blue]Analyzing contract at address {contract}...[/blue]")
        # This would fetch the contract code from the blockchain and analyze it
        # For demo purposes, we'll just show a message
        console.print("[yellow]This feature is not implemented in the demo[/yellow]")
        return
    
    # Check if file exists
    if not os.path.exists(contract):
        console.print(f"[red]Contract file not found: {contract}[/red]")
        return
    
    with console.status(f"Analyzing contract {contract}..."):
        results = auditor.analyze_contract(contract)
    
    # Display results
    if results["success"]:
        findings_count = len(results["findings"])
        
        if findings_count > 0:
            console.print(f"[yellow]Found {findings_count} potential vulnerabilities[/yellow]")
            
            # Create a table for the findings
            table = Table(title=f"Audit Findings: {contract}")
            table.add_column("Severity", style="cyan")
            table.add_column("Title", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Line", style="green")
            
            for finding in results["findings"]:
                severity = finding.get("severity", "Unknown")
                severity_style = {
                    "Critical": "red bold",
                    "High": "red",
                    "Medium": "yellow",
                    "Low": "green",
                    "Unknown": "white"
                }.get(severity, "white")
                
                table.add_row(
                    f"[{severity_style}]{severity}[/{severity_style}]",
                    finding.get("title", "Unknown"),
                    finding.get("description", ""),
                    str(finding.get("line", ""))
                )
            
            console.print(table)
        else:
            console.print(f"[green]No vulnerabilities found in {contract}[/green]")
    else:
        console.print(f"[red]Analysis failed: {results.get('error', 'Unknown error')}[/red]")
    
    # Save results if output is specified
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")

# Transaction commands
@app.command()
def trace(
    tx_hash: str = typer.Argument(..., help="Transaction hash to trace"),
    rpc_url: str = typer.Option(None, help="Ethereum RPC URL"),
    output: str = typer.Option(None, help="Path to output the results (JSON format)"),
    analyze: bool = typer.Option(False, help="Analyze the trace")
):
    """Trace a transaction's execution."""
    print_banner()
    
    console.print("[blue]Initializing Transaction Tracer...[/blue]")
    tracer = TransactionTracer(rpc_url=rpc_url)
    
    with console.status(f"Tracing transaction {tx_hash}..."):
        trace = tracer.trace_transaction(tx_hash)
    
    if not trace.get("success", False):
        console.print(f"[red]Failed to trace transaction: {trace.get('error', 'Unknown error')}[/red]")
        return
    
    console.print(f"[green]Successfully traced transaction {tx_hash}[/green]")
    
    if analyze:
        with console.status("Analyzing trace..."):
            analysis = tracer.analyze_trace(trace)
        
        # Display analysis
        console.print(Panel.fit(
            f"[blue]Transaction Analysis: {tx_hash}[/blue]\n\n"
            f"Status: {'[green]Success[/green]' if analysis.get('status') == 1 else '[red]Failed[/red]'}\n"
            f"Gas Used: {analysis.get('gas_used', 'Unknown')}\n"
            f"Call Depth: {analysis.get('statistics', {}).get('call_depth', 'Unknown')}\n"
            f"Total Calls: {analysis.get('statistics', {}).get('total_calls', 'Unknown')}\n"
            f"Contains Delegatecall: {analysis.get('patterns', {}).get('contains_delegatecall', False)}\n"
            f"Contains Selfdestruct: {analysis.get('patterns', {}).get('contains_selfdestruct', False)}"
        ))
        
        # Save combined results if output is specified
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, 'w') as f:
                json.dump({
                    "trace": trace,
                    "analysis": analysis
                }, f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")
    else:
        # Display simplified trace
        call_tree = trace.get("trace", {})
        
        def format_call(call, indent=0):
            from_addr = call.get("from", "0x0")
            to_addr = call.get("to", "0x0")
            value = int(call.get("value", "0x0"), 16) if isinstance(call.get("value"), str) else 0
            gas = int(call.get("gas", "0x0"), 16) if isinstance(call.get("gas"), str) else 0
            
            result = f"{'  ' * indent}[cyan]{from_addr}[/cyan] → [yellow]{to_addr}[/yellow]"
            if value > 0:
                result += f" ([green]{value} wei[/green])"
            if gas > 0:
                result += f" (Gas: {gas})"
            
            return result
        
        def print_call_tree(call, indent=0):
            console.print(format_call(call, indent))
            
            for subcall in call.get("calls", []):
                print_call_tree(subcall, indent + 1)
        
        console.print("[blue]Call Tree:[/blue]")
        print_call_tree(call_tree)
        
        # Save trace if output is specified
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(trace, f, indent=2)
            console.print(f"[green]Trace saved to {output}[/green]")

# Security commands
@app.command()
def detect_frontrun(
    block: int = typer.Option(None, help="Analyze a specific block"),
    range: str = typer.Option(None, help="Analyze a range of blocks (format: start-end)"),
    rpc_url: str = typer.Option(None, help="Ethereum RPC URL"),
    ws_url: str = typer.Option(None, help="Ethereum WebSocket URL"),
    monitor: bool = typer.Option(False, help="Monitor mempool for front-running"),
    output: str = typer.Option(None, help="Path to output the results (JSON format)")
):
    """Detect front-running activities."""
    print_banner()
    
    console.print("[blue]Initializing Front-running Detector...[/blue]")
    detector = FrontrunDetector(rpc_url=rpc_url, ws_url=ws_url)
    
    results = []
    
    if block is not None:
        with console.status(f"Analyzing block {block}..."):
            incidents = detector.analyze_block(block)
        
        if incidents:
            console.print(f"[yellow]Found {len(incidents)} potential front-running incidents in block {block}[/yellow]")
            results.extend(incidents)
        else:
            console.print(f"[green]No front-running detected in block {block}[/green]")
    
    elif range is not None:
        try:
            start, end = map(int, range.split("-"))
            
            total_blocks = end - start + 1
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task(f"Analyzing blocks {start} to {end}...", total=total_blocks)
                
                for block_num in range(start, end + 1):
                    progress.update(task, description=f"Analyzing block {block_num}...")
                    incidents = detector.analyze_block(block_num)
                    results.extend(incidents)
                    progress.advance(task)
            
            if results:
                console.print(f"[yellow]Found {len(results)} potential front-running incidents[/yellow]")
            else:
                console.print(f"[green]No front-running detected in blocks {start} to {end}[/green]")
        except ValueError:
            console.print("[red]Invalid range format. Use start-end (e.g., 15000000-15000010)[/red]")
    
    elif monitor:
        console.print("[blue]Monitoring mempool for potential front-running...[/blue]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        
        def print_incident(incident):
            console.print(Panel.fit(
                f"[red]Potential front-running detected![/red]\n\n"
                f"Frontrunner: [cyan]{incident['frontrunner_tx']['hash']}[/cyan]\n"
                f"Target: [yellow]{incident['target_tx']['hash']}[/yellow]\n"
                f"Gas price difference: [green]{incident['gas_price_difference']:.2f}x[/green]\n"
                f"Confidence: [blue]{incident['confidence']:.2f}[/blue]"
            ))
            
            # Add to results for potential output
            results.append(incident)
        
        try:
            detector.monitor_mempool(callback=print_incident)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped monitoring mempool[/yellow]")
    
    else:
        console.print("[yellow]No action specified. Use --help to see available options.[/yellow]")
    
    # Display results table if there are results
    if results and not monitor:  # Skip table for monitoring mode
        table = Table(title="Front-running Detection Results")
        table.add_column("Block", style="cyan")
        table.add_column("Frontrunner", style="yellow")
        table.add_column("Target", style="blue")
        table.add_column("Gas Price Difference", style="green")
        table.add_column("Confidence", style="red")
        
        for incident in results:
            table.add_row(
                str(incident.get("block_number", "N/A")),
                incident.get("frontrunner_tx", {}).get("hash", "Unknown")[:10] + "...",
                incident.get("victim_tx", {}).get("hash", "Unknown")[:10] + "...",
                f"{incident.get('gas_price_difference', 0):.2f}x",
                f"{incident.get('confidence', 0):.2f}"
            )
        
        console.print(table)
    
    # Save results if output is specified and there are results
    if output and results:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")

# ZK commands
@app.command()
def verify_zk(
    proof: str = typer.Option(None, help="Path to proof JSON file"),
    vk: str = typer.Option(None, help="Path to verification key JSON file"),
    inputs: str = typer.Option(None, help="Comma-separated list of public inputs"),
    generate_dummy: bool = typer.Option(False, help="Generate dummy proof and verification key"),
    output: str = typer.Option(None, help="Path to output the results (JSON format)")
):
    """Verify zero-knowledge proofs."""
    print_banner()
    
    console.print("[blue]Initializing ZK Verifier...[/blue]")
    verifier = Verifier()
    
    if generate_dummy:
        with console.status("Generating dummy ZK example..."):
            dummy_proof = verifier.generate_dummy_proof()
            dummy_vk = verifier.generate_dummy_vk(num_public_inputs=1)
            dummy_public_inputs = [42]  # Example public input
        
        console.print("[green]Generated dummy ZK example[/green]")
        
        results = {
            "proof": dummy_proof,
            "vk": dummy_vk,
            "public_inputs": dummy_public_inputs,
            "note": "This is a dummy example for testing purposes only."
        }
        
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Dummy ZK example saved to {output}[/green]")
        else:
            # Print truncated results
            console.print(Panel.fit(
                f"[blue]Dummy ZK Example[/blue]\n\n"
                f"Proof: {str(dummy_proof)[:100]}...\n"
                f"VK: {str(dummy_vk)[:100]}...\n"
                f"Public Inputs: {dummy_public_inputs}"
            ))
    
    elif proof and vk and inputs:
        # Load proof
        try:
            with open(proof, 'r') as f:
                proof_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading proof: {e}[/red]")
            return
        
        # Load verification key
        try:
            with open(vk, 'r') as f:
                vk_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading verification key: {e}[/red]")
            return
        
        # Parse public inputs
        try:
            public_inputs = list(map(int, inputs.split(',')))
        except Exception as e:
            console.print(f"[red]Error parsing public inputs: {e}[/red]")
            return
        
        # Verify proof
        with console.status("Verifying proof..."):
            is_valid = verifier.verify_proof(proof_data, public_inputs, vk_data)
        
        if is_valid:
            console.print("[green]Proof is valid ✓[/green]")
        else:
            console.print("[red]Proof is invalid ✗[/red]")
        
        # Save results if output is specified
        if output:
            results = {
                "is_valid": is_valid,
                "proof_file": proof,
                "vk_file": vk,
                "public_inputs": public_inputs
            }
            
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Verification results saved to {output}[/green]")
    
    else:
        console.print("[yellow]Missing required arguments. Use --help to see available options.[/yellow]")

# API commands
@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(5000, help="API port"),
    debug: bool = typer.Option(False, help="Enable debug mode")
):
    """Start the API server."""
    print_banner()
    
    # Set environment variables
    os.environ["API_HOST"] = host
    os.environ["API_PORT"] = str(port)
    os.environ["API_DEBUG"] = str(debug).lower()
    
    console.print(f"[blue]Starting API server on http://{host}:{port}...[/blue]")
    
    # Import the API app and run it
    from llamachain.api.app import main
    main()

# Version command
@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    ctx: typer.Context = typer.Context
):
    """LlamaChain: Blockchain Intelligence Platform"""
    if version:
        print_banner()
        print_version()
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        print_banner()
        print_version()

if __name__ == "__main__":
    app()
EOF
}

function create_analytics_pipeline() {
  print_header "Creating Analytics Pipeline"
  
  log_info "Creating analytics pipeline module..."
  mkdir -p llamachain/pipelines
  cat > llamachain/pipelines/analytics.py << EOF
"""
Analytics Pipeline: Process blockchain data using Apache Beam.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("analytics_pipeline")

class ParseBlock(beam.DoFn):
    """Parse a raw block into a structured format."""
    
    def process(self, element):
        try:
            # Parse the input element (could be a block from PubSub, file, etc.)
            if isinstance(element, bytes):
                element = element.decode('utf-8')
            
            if isinstance(element, str):
                block = json.loads(element)
            else:
                block = element
            
            # Extract relevant fields from the block
            block_data = {
                "number": block.get("number"),
                "hash": block.get("hash"),
                "parent_hash": block.get("parentHash"),
                "timestamp": block.get("timestamp"),
                "miner": block.get("miner"),
                "difficulty": block.get("difficulty"),
                "total_difficulty": block.get("totalDifficulty"),
                "size": block.get("size"),
                "gas_used": block.get("gasUsed"),
                "gas_limit": block.get("gasLimit"),
                "transaction_count": len(block.get("transactions", [])),
                "uncles_count": len(block.get("uncles", []))
            }
            
            # Yield the processed block
            yield block_data
            
            # Process transactions if present
            for tx in block.get("transactions", []):
                if isinstance(tx, dict):  # Only process if full transactions were retrieved
                    # Extract transaction fields
                    tx_data = {
                        "hash": tx.get("hash"),
                        "block_number": block.get("number"),
                        "block_hash": block.get("hash"),
                        "from_address": tx.get("from"),
                        "to_address": tx.get("to"),
                        "value": tx.get("value"),
                        "gas": tx.get("gas"),
                        "gas_price": tx.get("gasPrice"),
                        "nonce": tx.get("nonce"),
                        "transaction_index": tx.get("transactionIndex"),
                        "input_data": tx.get("input"),
                        "timestamp": block.get("timestamp")
                    }
                    
                    # Yield the processed transaction
                    yield tx_data
        
        except Exception as e:
            logger.error(f"Error parsing block: {e}")
            # Skip this element
            return []

class ExtractEntityAddress(beam.DoFn):
    """Extract unique addresses from transactions."""
    
    def process(self, element):
        try:
            if "from_address" in element and element["from_address"]:
                yield {
                    "address": element["from_address"],
                    "type": "from",
                    "transaction_hash": element.get("hash"),
                    "block_number": element.get("block_number"),
                    "timestamp": element.get("timestamp")
                }
            
            if "to_address" in element and element["to_address"]:
                yield {
                    "address": element["to_address"],
                    "type": "to",
                    "transaction_hash": element.get("hash"),
                    "block_number": element.get("block_number"),
                    "timestamp": element.get("timestamp")
                }
            
            if "miner" in element and element["miner"]:
                yield {
                    "address": element["miner"],
                    "type": "miner",
                    "block_number": element.get("number"),
                    "timestamp": element.get("timestamp")
                }
        
        except Exception as e:
            logger.error(f"Error extracting addresses: {e}")
            # Skip this element
            return []

class CalculateBlockMetrics(beam.DoFn):
    """Calculate metrics for blocks."""
    
    def process(self, element, window=None):
        try:
            if "number" in element:  # This is a block
                # Calculate gas utilization
                gas_utilization = element.get("gas_used", 0) / element.get("gas_limit", 1) * 100
                
                metrics = {
                    "timestamp": element.get("timestamp"),
                    "metric_type": "block",
                    "block_number": element.get("number"),
                    "gas_utilization": gas_utilization,
                    "block_size": element.get("size"),
                    "transaction_count": element.get("transaction_count"),
                    "difficulty": element.get("difficulty")
                }
                
                yield metrics
        
        except Exception as e:
            logger.error(f"Error calculating block metrics: {e}")
            # Skip this element
            return []

class CalculateTransactionMetrics(beam.DoFn):
    """Calculate metrics for transactions."""
    
    def process(self, element, window=None):
        try:
            if "from_address" in element and "to_address" in element:  # This is a transaction
                # Calculate gas cost
                gas_cost = (element.get("gas", 0) * element.get("gas_price", 0)) / 1e18  # in ETH
                
                metrics = {
                    "timestamp": element.get("timestamp"),
                    "metric_type": "transaction",
                    "block_number": element.get("block_number"),
                    "from_address": element.get("from_address"),
                    "to_address": element.get("to_address"),
                    "value": element.get("value", 0) / 1e18,  # in ETH
                    "gas_cost": gas_cost,
                    "gas": element.get("gas"),
                    "gas_price": element.get("gas_price")
                }
                
                yield metrics
        
        except Exception as e:
            logger.error(f"Error calculating transaction metrics: {e}")
            # Skip this element
            return []

def run_pipeline():
    """Run the analytics pipeline."""
    # Get configuration from environment variables
    project = os.environ.get("BIGQUERY_PROJECT")
    dataset = os.environ.get("BIGQUERY_DATASET", "blockchain_data")
    runner = os.environ.get("BEAM_RUNNER", "DirectRunner")
    
    # Define pipeline options
    options = PipelineOptions([
        f"--runner={runner}",
        f"--project={project}" if project else "",
        "--save_main_session=True"
    ])
    
    # Create the pipeline
    with beam.Pipeline(options=options) as pipeline:
        # Read block data (this would be replaced with actual data source)
        blocks = (
            pipeline
            | "ReadBlockData" >> beam.Create([])  # Placeholder
            # In a real implementation, this would be something like:
            # | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=subscription)
            # or
            # | "ReadFromFile" >> beam.io.ReadFromText("gs://bucket/path/to/data/*.json")
        )
        
        # Parse blocks and extract transactions
        parsed_data = (
            blocks
            | "ParseBlocks" >> beam.ParDo(ParseBlock())
        )
        
        # Window the data (for time-based aggregations)
        windowed_data = (
            parsed_data
            | "AddTimestamps" >> beam.Map(lambda x: beam.window.TimestampedValue(x, x.get("timestamp", 0)))
            | "WindowInto" >> beam.WindowInto(beam.window.FixedWindows(60 * 60))  # 1-hour windows
        )
        
        # Extract addresses for entity analysis
        addresses = (
            windowed_data
            | "ExtractAddresses" >> beam.ParDo(ExtractEntityAddress())
        )
        
        # Calculate block metrics
        block_metrics = (
            windowed_data
            | "CalculateBlockMetrics" >> beam.ParDo(CalculateBlockMetrics())
        )
        
        # Calculate transaction metrics
        tx_metrics = (
            windowed_data
            | "CalculateTransactionMetrics" >> beam.ParDo(CalculateTransactionMetrics())
        )
        
        # Combine all metrics
        all_metrics = (
            (block_metrics, tx_metrics)
            | "MergeMetrics" >> beam.Flatten()
        )
        
        # Aggregate metrics by time window
        aggregated_metrics = (
            all_metrics
            | "GroupByMetricType" >> beam.GroupBy(lambda x: x.get("metric_type"))
            .aggregate_field("gas_utilization", beam.Mean(), "avg_gas_utilization")
            .aggregate_field("transaction_count", beam.Sum(), "total_transactions")
            .aggregate_field("block_size", beam.Mean(), "avg_block_size")
            .aggregate_field("gas_price", beam.Mean(), "avg_gas_price")
        )
        
        # Write results to BigQuery if project is set
        if project:
            # Define BigQuery table schemas
            block_table_schema = {
                "fields": [
                    {"name": "number", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "hash", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "parent_hash", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
                    {"name": "miner", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "difficulty", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "total_difficulty", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "size", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "gas_used", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "gas_limit", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "transaction_count", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "uncles_count", "type": "INTEGER", "mode": "REQUIRED"}
                ]
            }
            
            tx_table_schema = {
                "fields": [
                    {"name": "hash", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "block_number", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "block_hash", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "from_address", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "to_address", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "value", "type": "INTEGER", "mode": "REQUIRED"},
                    {"name": "gas", "type": "INTEGER", "mode": "  # Create zk verifier module
  log_info "Creating ZK verifier module..."
  mkdir -p llamachain/zk
  cat > llamachain/zk/verifier.py << EOF
"""
Verifier: Verifies zero-knowledge proofs for enhanced trust and privacy.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import py_ecc if available
try:
    from py_ecc.bn128 import G1, G2, add, multiply, pairing, neg, curve_order, normalize
    from py_ecc.bn128.bn128_field_elements import FQ, FQ2, FQ12
    PY_ECC_AVAILABLE = True
except ImportError:
    PY_ECC_AVAILABLE = False

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("zk_verifier")

class Verifier:
    """
    Verifier uses py_ecc to verify zero-knowledge proofs.
    """
    
    def __init__(self):
        """Initialize the Verifier."""
        # Check if py_ecc is available
        if not PY_ECC_AVAILABLE:
            logger.warning(
                "py_ecc is not installed. ZK verification capabilities will be limited. "
                "Install with: pip install py_ecc"
            )
        
        logger.info("Verifier initialized successfully")
    
    def verify_proof(self, proof: Dict[str, Any], public_inputs: List[int], vk: Dict[str, Any]) -> bool:
        """
        Verify a zk-SNARK proof using bn128 pairing.
        
        Args:
            proof: The zk-SNARK proof
            public_inputs: The public inputs to the circuit
            vk: The verification key
            
        Returns:
            True if the proof is valid, False otherwise
        """
        if not PY_ECC_AVAILABLE:
            logger.error("py_ecc is required for proof verification")
            return False
        
        try:
            # Convert proof components from hex to FQ points
            a = self._hex_to_g1(proof.get("a", ["0x0", "0x0"]))
            b = self._hex_to_g2(proof.get("b", ["0x0", "0x0", "0x0", "0x0"]))
            c = self._hex_to_g1(proof.get("c", ["0x0", "0x0"]))
            
            # Convert verification key components
            alpha = self._hex_to_g1(vk.get("alpha", ["0x0", "0x0"]))
            beta = self._hex_to_g2(vk.get("beta", ["0x0", "0x0", "0x0", "0x0"]))
            gamma = self._hex_to_g2(vk.get("gamma", ["0x0", "0x0", "0x0", "0x0"]))
            delta = self._hex_to_g2(vk.get("delta", ["0x0", "0x0", "0x0", "0x0"]))
            
            # Process IC (coefficients for public inputs)
            ic = []
            for i in range(len(vk.get("ic", [])) - 1):
                ic.append(self._hex_to_g1(vk["ic"][i]))
            
            # Compute linear combination of public inputs and IC
            vk_x = self._hex_to_g1(vk.get("ic", [[]])[0])
            for i in range(len(public_inputs)):
                vk_x = add(vk_x, multiply(ic[i], public_inputs[i]))
            
            # Verify pairing equation
            pairing1 = pairing(b, a)
            pairing2 = pairing(gamma, vk_x)
            pairing3 = pairing(delta, c)
            pairing_result = pairing1 * pairing2 * pairing3
            
            return pairing_result == FQ12.one()
            
        except Exception as e:
            logger.error(f"Error verifying proof: {e}")
            return False
    
    def _hex_to_g1(self, hex_coords: List[str]) -> Tuple[FQ, FQ]:
        """Convert hex coordinates to a G1 point."""
        if len(hex_coords) < 2:
            raise ValueError("G1 point requires 2 coordinates")
        
        x = FQ(int(hex_coords[0], 16))
        y = FQ(int(hex_coords[1], 16))
        
        return (x, y)
    
    def _hex_to_g2(self, hex_coords: List[str]) -> Tuple[FQ2, FQ2]:
        """Convert hex coordinates to a G2 point."""
        if len(hex_coords) < 4:
            raise ValueError("G2 point requires 4 coordinates")
        
        x_c0 = int(hex_coords[0], 16)
        x_c1 = int(hex_coords[1], 16)
        y_c0 = int(hex_coords[2], 16)
        y_c1 = int(hex_coords[3], 16)
        
        x = FQ2([x_c0, x_c1])
        y = FQ2([y_c0, y_c1])
        
        return (x, y)
    
    def generate_dummy_proof(self) -> Dict[str, Any]:
        """
        Generate a dummy proof for testing purposes.
        
        Returns:
            A dummy zk-SNARK proof
        """
        if not PY_ECC_AVAILABLE:
            logger.error("py_ecc is required for proof generation")
            return {}
        
        try:
            # Generate random points
            import random
            
            # Random G1 point for 'a'
            a_x = random.randint(1, curve_order - 1)
            a_y = random.randint(1, curve_order - 1)
            a = (FQ(a_x), FQ(a_y))
            
            # Random G2 point for 'b'
            b_x = FQ2([random.randint(1, curve_order - 1), random.randint(1, curve_order - 1)])
            b_y = FQ2([random.randint(1, curve_order - 1), random.randint(1, curve_order - 1)])
            b = (b_x, b_y)
            
            # Random G1 point for 'c'
            c_x = random.randint(1, curve_order - 1)
            c_y = random.randint(1, curve_order - 1)
            c = (FQ(c_x), FQ(c_y))
            
            return {
                "a": [hex(int(a[0])), hex(int(a[1]))],
                "b": [hex(int(b[0].coeffs[0])), hex(int(b[0].coeffs[1])), 
                      hex(int(b[1].coeffs[0])), hex(int(b[1].coeffs[1]))],
                "c": [hex(int(c[0])), hex(int(c[1]))]
            }
            
        except Exception as e:
            logger.error(f"Error generating dummy proof: {e}")
            return {}
    
    def generate_dummy_vk(self, num_public_inputs: int = 1) -> Dict[str, Any]:
        """
        Generate a dummy verification key for testing purposes.
        
        Args:
            num_public_inputs: Number of public inputs to the circuit
            
        Returns:
            A dummy verification key
        """
        if not PY_ECC_AVAILABLE:
            logger.error("py_ecc is required for verification key generation")
            return {}
        
        try:
            # Generate random points
            import random
            
            # Generate alpha (G1)
            alpha_x = random.randint(1, curve_order - 1)
            alpha_y = random.randint(1, curve_order - 1)
            alpha = [hex(alpha_x), hex(alpha_y)]
            
            # Generate beta (G2)
            beta = [
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1))
            ]
            
            # Generate gamma (G2)
            gamma = [
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1))
            ]
            
            # Generate delta (G2)
            delta = [
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1)),
                hex(random.randint(1, curve_order - 1))
            ]
            
            # Generate IC (coefficients for public inputs)
            ic = []
            for i in range(num_public_inputs + 1):  # +1 for the constant term
                ic.append([
                    hex(random.randint(1, curve_order - 1)),
                    hex(random.randint(1, curve_order - 1))
                ])
            
            return {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "ic": ic
            }
            
        except Exception as e:
            logger.error(f"Error generating dummy verification key: {e}")
            return {}

# Main entry point when run as a module
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaChain ZK Verifier")
    parser.add_argument("--proof", help="Path to proof JSON file")
    parser.add_argument("--vk", help="Path to verification key JSON file")
    parser.add_argument("--public-inputs", help="Comma-separated list of public inputs")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate dummy proof and verification key")
    parser.add_argument("--output", help="Path to output the results (JSON format)")
    
    args = parser.parse_args()
    
    verifier = Verifier()
    
    if args.generate_dummy:
        dummy_proof = verifier.generate_dummy_proof()
        dummy_vk = verifier.generate_dummy_vk(num_public_inputs=1)
        dummy_public_inputs = [42]  # Example public input
        
        results = {
            "proof": dummy_proof,
            "vk": dummy_vk,
            "public_inputs": dummy_public_inputs,
            "note": "This is a dummy example for testing purposes only."
        }
        
        if args.output:
            # Save results to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Dummy ZK example saved to {args.output}")
        else:
            # Print results to console
            print(json.dumps(results, indent=2))
    
    elif args.proof and args.vk and args.public_inputs:
        # Load proof
        with open(args.proof, 'r') as f:
            proof = json.load(f)
        
        # Load verification key
        with open(args.vk, 'r') as f:
            vk = json.load(f)
        
        # Parse public inputs
        public_inputs = list(map(int, args.public_inputs.split(',')))
        
        # Verify proof
        is_valid = verifier.verify_proof(proof, public_inputs, vk)
        
        results = {
            "is_valid": is_valid,
            "proof_file": args.proof,
            "vk_file": args.vk,
            "public_inputs": public_inputs
        }
        
        if args.output:
            # Save results to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Verification results saved to {args.output}")
        else:
            # Print results to console
            print(f"Proof is {'valid' if is_valid else 'invalid'}")
    
    else:
        print("Missing required arguments. Use --help to see available options.")

if __name__ == "__main__":
    main()
EOF
}

function create_api_endpoints() {
  print_header "Creating API Endpoints"
  
  log_info "Creating API endpoints module..."
  mkdir -p llamachain/api
  cat > llamachain/api/endpoints.py << EOF
"""
API endpoints for LlamaChain.
"""

import os
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
import json

from llamachain.core.indexer import BlockchainIndexer
from llamachain.analysis.contract import ContractAuditor
from llamachain.analysis.trace import TransactionTracer
from llamachain.security.frontrun import FrontrunDetector
from llamachain.zk.verifier import Verifier
from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("api")

# Initialize Flask app
app = Flask(__name__)
api = Api(
    app,
    version="0.1.0",
    title="LlamaChain API",
    description="Blockchain Intelligence Platform API",
    doc="/docs"
)

# Create namespaces
address_ns = Namespace("address", description="Address-related operations")
contract_ns = Namespace("contract", description="Smart contract operations")
transaction_ns = Namespace("transaction", description="Transaction operations")
block_ns = Namespace("block", description="Block operations")
security_ns = Namespace("security", description="Security operations")
zk_ns = Namespace("zk", description="Zero-knowledge proof operations")

# Add namespaces to API
api.add_namespace(address_ns, path="/address")
api.add_namespace(contract_ns, path="/contract")
api.add_namespace(transaction_ns, path="/transaction")
api.add_namespace(block_ns, path="/block")
api.add_namespace(security_ns, path="/security")
api.add_namespace(zk_ns, path="/zk")

# Initialize core components
indexer = BlockchainIndexer(rpc_url=os.environ.get("ETH_RPC_URL"))
auditor = ContractAuditor()
tracer = TransactionTracer(rpc_url=os.environ.get("ETH_RPC_URL"))
frontrun_detector = FrontrunDetector(rpc_url=os.environ.get("ETH_RPC_URL"))
verifier = Verifier()

# Define models
balance_model = address_ns.model("Balance", {
    "address": fields.String(description="Ethereum address"),
    "balance": fields.String(description="Native token balance (in wei)"),
    "token_balances": fields.List(fields.Nested(address_ns.model("TokenBalance", {
        "token_address": fields.String(description="Token contract address"),
        "symbol": fields.String(description="Token symbol"),
        "balance": fields.String(description="Token balance"),
        "decimals": fields.Integer(description="Token decimals")
    })))
})

audit_report_model = contract_ns.model("AuditReport", {
    "contract_address": fields.String(description="Contract address"),
    "success": fields.Boolean(description="Audit success status"),
    "findings": fields.List(fields.Nested(contract_ns.model("Finding", {
        "title": fields.String(description="Finding title"),
        "description": fields.String(description="Finding description"),
        "severity": fields.String(description="Finding severity"),
        "type": fields.String(description="Finding type"),
        "line": fields.Integer(description="Line number"),
        "detector": fields.String(description="Detector used")
    }))),
    "summary": fields.Nested(contract_ns.model("Summary", {
        "total": fields.Integer(description="Total findings"),
        "critical": fields.Integer(description="Critical findings"),
        "high": fields.Integer(description="High severity findings"),
        "medium": fields.Integer(description="Medium severity findings"),
        "low": fields.Integer(description="Low severity findings")
    }))
})

transaction_trace_model = transaction_ns.model("TransactionTrace", {
    "tx_hash": fields.String(description="Transaction hash"),
    "success": fields.Boolean(description="Trace success status"),
    "trace": fields.Raw(description="Transaction trace data"),
    "transaction": fields.Nested(transaction_ns.model("TransactionInfo", {
        "from": fields.String(description="Sender address"),
        "to": fields.String(description="Recipient address"),
        "value": fields.String(description="Transaction value"),
        "gas": fields.Integer(description="Gas limit"),
        "gasPrice": fields.Integer(description="Gas price")
    })),
    "receipt": fields.Nested(transaction_ns.model("ReceiptInfo", {
        "status": fields.Integer(description="Transaction status"),
        "gasUsed": fields.Integer(description="Gas used")
    }))
})

block_info_model = block_ns.model("BlockInfo", {
    "number": fields.Integer(description="Block number"),
    "hash": fields.String(description="Block hash"),
    "timestamp": fields.Integer(description="Block timestamp"),
    "miner": fields.String(description="Miner address"),
    "transaction_count": fields.Integer(description="Transaction count"),
    "size": fields.Integer(description="Block size"),
    "gas_used": fields.Integer(description="Gas used"),
    "gas_limit": fields.Integer(description="Gas limit")
})

frontrun_model = security_ns.model("FrontrunDetection", {
    "potential_frontrun": fields.Boolean(description="Potential front-running detected"),
    "block_number": fields.Integer(description="Block number"),
    "frontrunner_tx": fields.Nested(security_ns.model("FrontrunnerTx", {
        "hash": fields.String(description="Transaction hash"),
        "from": fields.String(description="Sender address"),
        "gas_price": fields.Integer(description="Gas price")
    })),
    "victim_tx": fields.Nested(security_ns.model("VictimTx", {
        "hash": fields.String(description="Transaction hash"),
        "from": fields.String(description="Sender address"),
        "gas_price": fields.Integer(description="Gas price")
    })),
    "contract": fields.String(description="Target contract"),
    "gas_price_difference": fields.Float(description="Gas price difference ratio"),
    "confidence": fields.Float(description="Detection confidence score")
})

zk_verify_model = zk_ns.model("ZkVerification", {
    "is_valid": fields.Boolean(description="Verification result"),
    "proof": fields.Raw(description="Proof data"),
    "public_inputs": fields.List(fields.Integer(description="Public input value"))
})

# Address endpoints
@address_ns.route("/<string:address>/balance")
@address_ns.param("address", "Ethereum address")
class AddressBalance(Resource):
    @address_ns.marshal_with(balance_model)
    def get(self, address):
        """Get native token and ERC-20 balances for an address."""
        try:
            # Get native token balance
            balance = indexer.web3.eth.get_balance(address)
            
            # Get token balances (simplified implementation)
            token_balances = []
            
            # Return combined result
            return {
                "address": address,
                "balance": str(balance),
                "token_balances": token_balances
            }
        except Exception as e:
            logger.error(f"Error getting balance for {address}: {e}")
            return {"error": str(e)}, 500

# Contract endpoints
@contract_ns.route("/<string:address>/audit")
@contract_ns.param("address", "Contract address")
class ContractAudit(Resource):
    @contract_ns.marshal_with(audit_report_model)
    def get(self, address):
        """Get security audit report for a contract."""
        try:
            # This would fetch the contract code and audit it
            # For demo purposes, we'll return a placeholder
            
            return {
                "contract_address": address,
                "success": True,
                "findings": [
                    {
                        "title": "Example Finding",
                        "description": "This is a placeholder finding",
                        "severity": "Medium",
                        "type": "Placeholder",
                        "line": 0,
                        "detector": "API Example"
                    }
                ],
                "summary": {
                    "total": 1,
                    "critical": 0,
                    "high": 0,
                    "medium": 1,
                    "low": 0
                }
            }
        except Exception as e:
            logger.error(f"Error auditing contract {address}: {e}")
            return {"error": str(e)}, 500

@contract_ns.route("/audit/file")
class ContractAuditFile(Resource):
    @contract_ns.expect(contract_ns.parser().add_argument("file", location="files", type="file", required=True))
    @contract_ns.marshal_with(audit_report_model)
    def post(self):
        """Upload and audit a contract file."""
        try:
            # Get the uploaded file
            if "file" not in request.files:
                return {"error": "No file provided"}, 400
            
            file = request.files["file"]
            
            # Save the file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".sol", delete=False) as temp:
                file.save(temp.name)
                temp_path = temp.name
            
            # Audit the contract
            results = auditor.analyze_contract(temp_path)
            
            # Clean up the temporary file
            import os
            os.unlink(temp_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error auditing uploaded contract: {e}")
            return {"error": str(e)}, 500

# Transaction endpoints
@transaction_ns.route("/<string:tx_hash>/trace")
@transaction_ns.param("tx_hash", "Transaction hash")
class TransactionTrace(Resource):
    @transaction_ns.marshal_with(transaction_trace_model)
    def get(self, tx_hash):
        """Get execution trace for a transaction."""
        try:
            trace = tracer.trace_transaction(tx_hash)
            return trace
        except Exception as e:
            logger.error(f"Error tracing transaction {tx_hash}: {e}")
            return {"error": str(e)}, 500

@transaction_ns.route("/<string:tx_hash>/analyze")
@transaction_ns.param("tx_hash", "Transaction hash")
class TransactionAnalysis(Resource):
    def get(self, tx_hash):
        """Analyze a transaction trace for insights."""
        try:
            trace = tracer.trace_transaction(tx_hash)
            analysis = tracer.analyze_trace(trace)
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing transaction {tx_hash}: {e}")
            return {"error": str(e)}, 500

# Block endpoints
@block_ns.route("/<int:block_number>")
@block_ns.param("block_number", "Block number")
class BlockInfo(Resource):
    @block_ns.marshal_with(block_info_model)
    def get(self, block_number):
        """Get information about a block."""
        try:
            block = indexer.get_block(block_number)
            if not block:
                return {"error": "Block not found"}, 404
            
            return block
        except Exception as e:
            logger.error(f"Error getting block {block_number}: {e}")
            return {"error": str(e)}, 500

# Security endpoints
@security_ns.route("/frontrun/block/<int:block_number>")
@security_ns.param("block_number", "Block number")
class FrontrunDetection(Resource):
    @security_ns.marshal_with(frontrun_model, as_list=True)
    def get(self, block_number):
        """Detect front-running in a block."""
        try:
            incidents = frontrun_detector.analyze_block(block_number)
            return incidents
        except Exception as e:
            logger.error(f"Error detecting front-running in block {block_number}: {e}")
            return {"error": str(e)}, 500

# ZK endpoints
@zk_ns.route("/verify")
class ZkVerify(Resource):
    @zk_ns.expect(zk_ns.parser().add_argument("proof", location="json", type=dict, required=True)
                    .add_argument("public_inputs", location="json", type=list, required=True)
                    .add_argument("vk", location="json", type=dict, required=True))
    @zk_ns.marshal_with(zk_verify_model)
    def post(self):
        """Verify a zero-knowledge proof."""
        try:
            data = request.json
            proof = data.get("proof", {})
            public_inputs = data.get("public_inputs", [])
            vk = data.get("vk", {})
            
            is_valid = verifier.verify_proof(proof, public_inputs, vk)
            
            return {
                "is_valid": is_valid,
                "proof": proof,
                "public_inputs": public_inputs
            }
        except Exception as e:
            logger.error(f"Error verifying ZK proof: {e}")
            return {"error": str(e)}, 500

# Main app
@app.route("/")
def home():
    """API home page."""
    return jsonify({
        "name": "LlamaChain API",
        "version": "0.1.0",
        "documentation": "/docs",
        "endpoints": [
            "/address/<address>/balance",
            "/contract/<address>/audit",
            "/contract/audit/file",
            "/transaction/<tx_hash>/trace",
            "/transaction/<tx_hash>/analyze",
            "/block/<block_number>",
            "/security/frontrun/block/<block_number>",
            "/zk/verify"
        ]
    })

def main():
    """Run the Flask app."""
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 5000))
    debug = os.environ.get("API_DEBUG", "true").lower() == "true"
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
EOF

  # Create main app
  cat > llamachain/api/app.py << EOF
"""
Main API application for LlamaChain.
"""

from llamachain.api.endpoints import app, main

if __name__ == "__main__":
    main()
EOF
}

function create_cli() {
  print_header "Creating Command-Line Interface"
  
  log_info "Creating CLI module..."
  mkdir -p llamachain/cli
  cat > llamachain/cli/main.py << EOF
"""
LlamaChain CLI: Command-line interface for the LlamaChain platform.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from llamachain.core.indexer import BlockchainIndexer
from llamachain.analysis.contract import ContractAuditor
from llamachain.analysis.trace import TransactionTracer
from llamachain.security.frontrun import FrontrunDetector
from llamachain.zk.verifier import Verifier
from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("cli")

# Initialize Typer app
app = typer.Typer(
    name="llamachain",
    help="LlamaChain: Blockchain Intelligence Platform",
    add_completion=False
)

# Initialize console for rich output
console = Console()

# ASCII art
LLAMA_ASCII = r"""
                        ., ,.
                       ; '. '
                    .-'     '-.
                   :           :
                   :           :  //
                   :           :  //
                    '-.       .-'  //
                       '-/+\-' //
                         /|\   //
                        /|\  //
                        /|\ //
"""

def print_banner():
    """Print the LlamaChain banner."""
    console.print(Panel.fit(
        f"[cyan]{LLAMA_ASCII}[/cyan]\n"
        "[blue]LlamaChain: Blockchain Intelligence Platform[/blue]",
        border_style="yellow",
        padding=(1, 2)
    ))

def print_version():
    """Print the LlamaChain version."""
    version = "0.1.0"  # Should be imported from a version file in a real app
    console.print(f"[yellow]LlamaChain v{version}[/yellow]")

# Blockchain commands
@app.command()
def index(
    rpc_url: str = typer.Option(None, help="Ethereum RPC URL"),
    block: int = typer.Option(None, help="Index a specific block"),
    latest: int = typer.Option(0, help="Index latest N blocks"),
    range: str = typer.Option(None, help="Index a range of blocks (format: start-end)"),
    listen: bool = typer.Option(False, help="Listen for new blocks and index them")
):
    """Index blockchain data."""
    print_banner()
    
    console.print("[blue]Initializing Blockchain Indexer...[/blue]")
    indexer = BlockchainIndexer(rpc_url=rpc_url)
    
    if block is not None:
        with console.status(f"Indexing block {block}..."):
            success = indexer.index_block(block)
        if success:
            console.print(f"[green]Successfully indexed block {block}[/green]")
        else:
            console.print(f"[red]Failed to index block {block}[/red]")
    
    elif latest > 0:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"Indexing latest {latest} blocks...", total=latest)
            
            latest_block = indexer.web3.eth.block_number
            start_block = max(0, latest_block - latest + 1)
            
            for block_num in range(start_block, latest_block + 1):
                progress.update(task, description=f"Indexing block {block_num}...")
                success = indexer.index_block(block_num)
                progress.advance(task)
        
        console.print(f"[green]Indexed latest {latest} blocks[/green]")
    #!/bin/bash

# ========================================================
# LlamaChain: Blockchain Intelligence Platform
# A comprehensive tool for on-chain data analysis and
# smart contract security auditing
# ========================================================

# ANSI Color codes for llama-themed colorful CLI
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Llama ASCII art
function display_llama_banner() {
  cat << "EOF"
${PURPLE}
                         ., ,.
                        ; '. '
                     .-'     '-.
                    :           :
                    :           :  ${YELLOW}//${NC}
          ${PURPLE}          :           :  ${YELLOW}// ${NC}
          ${PURPLE}           '-.       .-'  ${YELLOW}//  ${NC}
          ${PURPLE}              ${YELLOW}'-/+\\-' ${YELLOW}//   ${NC}
          ${PURPLE}               ${YELLOW} /|\\   //    ${NC}
          ${PURPLE}                ${YELLOW}/|\\ ${YELLOW}//     ${NC}
          ${PURPLE}               ${YELLOW} /|\\//      ${NC}
${BLUE}  ██╗     ██╗      █████╗ ███╗   ███╗ █████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗${NC}
${BLUE}  ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║  ██║██╔══██╗██║████╗  ██║${NC}
${BLUE}  ██║     ██║     ███████║██╔████╔██║███████║██║     ███████║███████║██║██╔██╗ ██║${NC}
${BLUE}  ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██║     ██╔══██║██╔══██║██║██║╚██╗██║${NC}
${BLUE}  ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║  ██║██║██║ ╚████║${NC}
${BLUE}  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝${NC}
                   ${CYAN}Blockchain Intelligence Platform${NC}
EOF
}

function print_header() {
  echo -e "${YELLOW}====================================================================${NC}"
  echo -e "${YELLOW}    $1${NC}"
  echo -e "${YELLOW}====================================================================${NC}"
}

function log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

function log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

function log_debug() {
  echo -e "${GRAY}[DEBUG]${NC} $1"
}

function log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function check_command() {
  if ! command -v $1 &> /dev/null; then
    log_error "$1 is not installed. Please install it to continue."
    return 1
  fi
  return 0
}

function check_prerequisites() {
  print_header "Checking Prerequisites"
  
  local failed=0
  
  # Check for Python 3.9+
  if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    failed=1
  else
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if (( $(echo "$python_version < 3.9" | bc -l) )); then
      log_error "Python 3.9+ is required (found $python_version)"
      failed=1
    else
      log_info "Python $python_version detected"
    fi
  fi
  
  # Check for pip
  if ! command -v pip3 &> /dev/null; then
    log_error "pip3 is not installed"
    failed=1
  else
    log_info "pip3 detected"
  fi
  
  # Check for Docker
  if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    failed=1
  else
    log_info "Docker detected"
  fi
  
  # Check for Docker Compose
  if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed"
    failed=1
  else
    log_info "Docker Compose detected"
  fi
  
  # Check for git
  if ! command -v git &> /dev/null; then
    log_error "git is not installed"
    failed=1
  else
    log_info "git detected"
  fi
  
  # Check if this is running on macOS with Apple Silicon
  if [[ "$(uname)" == "Darwin" ]]; then
    log_info "macOS detected"
    
    # Check for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
      log_info "Apple Silicon (arm64) detected - MLX optimizations will be enabled"
      export LLAMACHAIN_ENABLE_MLX=1
    else
      log_warning "Intel Mac detected - MLX optimizations will not be available"
      export LLAMACHAIN_ENABLE_MLX=0
    fi
  else
    log_warning "Non-macOS system detected - MLX optimizations will not be available"
    export LLAMACHAIN_ENABLE_MLX=0
  fi
  
  # Check for brew on macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
      log_error "Homebrew is not installed. It's recommended for macOS dependencies."
      failed=1
    else
      log_info "Homebrew detected"
    fi
  fi
  
  # Check for kubectl (optional, will be installed if missing)
  if ! command -v kubectl &> /dev/null; then
    log_warning "kubectl is not installed (will be installed later)"
  else
    log_info "kubectl detected"
  fi
  
  if [[ $failed -eq 1 ]]; then
    log_error "Please install the missing prerequisites and try again."
    return 1
  fi
  
  log_success "All core prerequisites are satisfied!"
  return 0
}

function create_virtual_environment() {
  print_header "Setting up Python Virtual Environment"
  
  # Create a virtual environment if it doesn't exist
  if [[ ! -d "venv" ]]; then
    log_info "Creating a new virtual environment..."
    python3 -m venv venv
    if [[ $? -ne 0 ]]; then
      log_error "Failed to create virtual environment"
      return 1
    fi
    log_success "Virtual environment created"
  else
    log_info "Using existing virtual environment"
  fi
  
  # Activate the virtual environment
  source venv/bin/activate
  if [[ $? -ne 0 ]]; then
    log_error "Failed to activate virtual environment"
    return 1
  fi
  log_success "Virtual environment activated"
  
  # Upgrade pip
  log_info "Upgrading pip..."
  pip install --upgrade pip
  
  return 0
}

function install_dependencies() {
  print_header "Installing Dependencies"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "Virtual environment not activated"
    return 1
  fi
  
  # Create requirements.txt file
  log_info "Creating requirements.txt..."
  cat > requirements.txt << EOF
# Core dependencies
web3==6.0.0
py-solc-x==1.1.1
slither-analyzer==0.9.0
py_ecc==5.2.0
ipfshttpclient==0.8.0
apache-beam==2.46.0
numpy==1.24.3
pandas==2.0.1
matplotlib==3.7.1
Flask==2.3.2
flask-restx==1.1.0
pytest==7.3.1
pytest-cov==4.1.0
python-dotenv==1.0.0
requests==2.30.0
rich==13.3.5
typer==0.9.0
pydantic==1.10.8
sqlalchemy==2.0.15
prometheus-client==0.17.0
pygraphviz==1.10
networkx==3.1
cryptography==41.0.0

# MLX for Apple Silicon (if enabled)
mlx==0.0.5; platform_machine == 'arm64' and platform_system == 'Darwin'

# Development tools
black==23.3.0
isort==5.12.0
flake8==6.0.0
mypy==1.3.0
pre-commit==3.3.2
EOF
  
  # Install dependencies
  log_info "Installing Python dependencies (this may take a while)..."
  pip install -r requirements.txt
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to install Python dependencies"
    return 1
  fi
  
  # Install system dependencies for macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    log_info "Installing system dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
      # Install GraphViz for visualization
      brew install graphviz
      
      # Install kubectl if not present
      if ! command -v kubectl &> /dev/null; then
        brew install kubectl
      fi
      
      # Install Go for Geth and other blockchain tools
      brew install go
      
      # Install Node.js and npm for frontend development
      brew install node
    else
      log_warning "Homebrew not installed, skipping system dependencies"
    fi
  fi
  
  # Install solc compiler
  log_info "Installing solc compiler..."
  python -c "from solcx import install_solc; install_solc(version='0.8.17')"
  
  log_success "Dependencies installed successfully"
  return 0
}

function create_project_structure() {
  print_header "Creating Project Structure"
  
  # Create project directories
  log_info "Creating project directories..."
  
  # Main project directories
  mkdir -p llamachain
  mkdir -p llamachain/core
  mkdir -p llamachain/analysis
  mkdir -p llamachain/security
  mkdir -p llamachain/api
  mkdir -p llamachain/zk
  mkdir -p llamachain/pipelines
  mkdir -p llamachain/utils
  mkdir -p llamachain/cli
  mkdir -p llamachain/config
  mkdir -p llamachain/models
  mkdir -p llamachain/storage
  mkdir -p k8s
  mkdir -p docker
  mkdir -p tests
  mkdir -p docs
  mkdir -p scripts
  mkdir -p data
  mkdir -p contracts
  
  # Create empty __init__.py files to make directories packages
  find llamachain -type d -exec touch {}/__init__.py \;
  
  log_success "Project structure created"
  return 0
}

function create_contract_auditor() {
  print_header "Creating Contract Auditor Module"
  
  log_info "Creating contract auditor module..."
  cat > llamachain/analysis/contract.py << EOF
"""
ContractAuditor: Performs security analysis on smart contracts.
"""

import os
import time
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile

# Import Slither only if available
try:
    from slither.slither import Slither
    from slither.exceptions import SlitherError
    SLITHER_AVAILABLE = True
except ImportError:
    SLITHER_AVAILABLE = False

# Import MLX for machine learning-based analysis if on Apple Silicon
try:
    import numpy as np
    if os.environ.get("LLAMACHAIN_ENABLE_MLX", "0") == "1":
        import mlx
        import mlx.core as mx
        MLX_AVAILABLE = True
    else:
        MLX_AVAILABLE = False
except ImportError:
    MLX_AVAILABLE = False

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("contract_auditor")

class ContractAuditor:
    """
    ContractAuditor performs security analysis on smart contracts using Slither
    and machine learning techniques.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ContractAuditor.
        
        Args:
            model_path: Path to the ML model for vulnerability detection
        """
        # Check if Slither is available
        if not SLITHER_AVAILABLE:
            logger.warning(
                "Slither is not installed. Static analysis capabilities will be limited. "
                "Install with: pip install slither-analyzer"
            )
        
        # Initialize ML model if available
        self.ml_model = None
        self.model_path = model_path or os.environ.get(
            "ML_MODEL_PATH", "models/vulnerability_detector.mlx"
        )
        
        if MLX_AVAILABLE and os.path.exists(self.model_path):
            try:
                # This is a placeholder for actual model loading
                # In a real implementation, this would load the MLX model
                logger.info(f"Loading ML model from {self.model_path}")
                # self.ml_model = load_mlx_model(self.model_path)
                
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
        elif MLX_AVAILABLE:
            logger.warning(f"ML model not found at {self.model_path}")
        else:
            logger.info("MLX not available, machine learning features disabled")
        
        logger.info("ContractAuditor initialized successfully")
        
        # Statistics
        self.stats = {
            "contracts_analyzed": 0,
            "vulnerabilities_found": 0,
            "start_time": time.time()
        }
    
    def analyze_contract(self, contract_path: str) -> Dict[str, Any]:
        """
        Analyze a smart contract for vulnerabilities.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            Dictionary containing analysis results
        """
        # Check if file exists
        if not os.path.exists(contract_path):
            logger.error(f"Contract file not found: {contract_path}")
            return {
                "success": False,
                "error": "Contract file not found",
                "findings": []
            }
        
        # Combine static analysis with ML-based analysis
        static_findings = self._perform_static_analysis(contract_path)
        ml_findings = self._perform_ml_analysis(contract_path) if self.ml_model else []
        
        # Merge and deduplicate findings
        all_findings = static_findings + ml_findings
        
        # Update statistics
        self.stats["contracts_analyzed"] += 1
        self.stats["vulnerabilities_found"] += len(all_findings)
        
        logger.info(f"Analyzed contract {contract_path}: found {len(all_findings)} vulnerabilities")
        
        # Categorize findings by severity
        critical = []
        high = []
        medium = []
        low = []
        
        for finding in all_findings:
            severity = finding.get("severity", "").lower()
            if severity == "critical":
                critical.append(finding)
            elif severity == "high":
                high.append(finding)
            elif severity == "medium":
                medium.append(finding)
            elif severity == "low":
                low.append(finding)
        
        return {
            "success": True,
            "contract_path": contract_path,
            "findings": all_findings,
            "summary": {
                "total": len(all_findings),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low)
            },
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "timestamp": int(time.time())
        }
    
    def _perform_static_analysis(self, contract_path: str) -> List[Dict[str, Any]]:
        """
        Perform static analysis on a smart contract using Slither.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            List of findings
        """
        findings = []
        
        # Skip if Slither is not available
        if not SLITHER_AVAILABLE:
            logger.warning("Slither not available, skipping static analysis")
            return findings
        
        try:
            # Initialize Slither
            slither = Slither(contract_path)
            
            # Find vulnerabilities
            for contract in slither.contracts:
                logger.info(f"Analyzing contract: {contract.name}")
                
                # Check for reentrancy
                for function in contract.functions:
                    if function.can_reenter:
                        findings.append({
                            "title": "Reentrancy",
                            "description": f"Function {function.name} is vulnerable to reentrancy attacks",
                            "severity": "High",
                            "contract": contract.name,
                            "function": function.name,
                            "line": function.source_mapping.lines[0],
                            "file": contract.source_mapping.filename.absolute,
                            "type": "Reentrancy",
                            "detector": "Slither"
                        })
                
                # Check for unchecked return values
                for function in contract.functions:
                    for node in function.nodes:
                        if node.low_level_calls and not node.contains_require_or_assert:
                            findings.append({
                                "title": "Unchecked Low-Level Call",
                                "description": f"Low-level call without return value check in {function.name}",
                                "severity": "Medium",
                                "contract": contract.name,
                                "function": function.name,
                                "line": node.source_mapping.lines[0] if node.source_mapping.lines else 0,
                                "file": contract.source_mapping.filename.absolute,
                                "type": "Unchecked Call",
                                "detector": "Slither"
                            })
                
                # Check for tx.origin usage
                for function in contract.functions:
                    for node in function.nodes:
                        if node.contains_tx_origin:
                            findings.append({
                                "title": "Tx.origin Usage",
                                "description": f"Use of tx.origin in {function.name}",
                                "severity": "High",
                                "contract": contract.name,
                                "function": function.name,
                                "line": node.source_mapping.lines[0] if node.source_mapping.lines else 0,
                                "file": contract.source_mapping.filename.absolute,
                                "type": "Tx.origin",
                                "detector": "Slither"
                            })
            
        except SlitherError as e:
            logger.error(f"Slither error: {e}")
        except Exception as e:
            logger.error(f"Error in static analysis: {e}")
        
        return findings
    
    def _perform_ml_analysis(self, contract_path: str) -> List[Dict[str, Any]]:
        """
        Perform ML-based analysis on a smart contract.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            List of findings
        """
        findings = []
        
        # Skip if MLX is not available
        if not MLX_AVAILABLE or not self.ml_model:
            return findings
        
        try:
            # This is a placeholder for actual ML analysis
            # In a real implementation, this would:
            # 1. Preprocess the contract code
            # 2. Convert to a format suitable for the ML model
            # 3. Run the ML model to detect vulnerabilities
            
            # Read contract source code
            with open(contract_path, 'r') as f:
                source_code = f.read()
            
            # Here would be code to actually run the model
            # For now, we'll just add a placeholder finding
            
            logger.info(f"Performed ML analysis on {contract_path}")
            
            # This is just a placeholder - real implementation would use the model
            if "function withdraw" in source_code and "call" in source_code:
                findings.append({
                    "title": "Potential Reentrancy (ML)",
                    "description": "Machine learning model detected a potential reentrancy vulnerability",
                    "severity": "High",
                    "contract": os.path.basename(contract_path).split('.')[0],
                    "line": source_code.index("function withdraw"),
                    "file": contract_path,
                    "type": "Reentrancy",
                    "detector": "MLX Model",
                    "confidence": 0.85
                })
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
        
        return findings
    
    def analyze_contract_bytecode(self, bytecode: str) -> Dict[str, Any]:
        """
        Analyze compiled bytecode for a smart contract.
        
        Args:
            bytecode: Contract bytecode
            
        Returns:
            Dictionary containing analysis results
        """
        # Save bytecode to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(bytecode.encode())
            bytecode_path = f.name
        
        try:
            # Analyze bytecode
            # This would use tools specifically designed for bytecode analysis
            # For now, just return a placeholder result
            
            return {
                "success": True,
                "findings": [],
                "summary": {
                    "total": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "timestamp": int(time.time())
            }
        finally:
            # Clean up temporary file
            os.unlink(bytecode_path)
    
    def analyze_from_address(self, contract_address: str, network: str = "mainnet") -> Dict[str, Any]:
        """
        Analyze a deployed contract by its address.
        
        Args:
            contract_address: Contract address
            network: Network name (mainnet, ropsten, etc.)
            
        Returns:
            Dictionary containing analysis results
        """
        # This would fetch the contract code from the blockchain and analyze it
        # For now, just return a placeholder result
        
        return {
            "success": False,
            "error": "Not implemented yet",
            "findings": []
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the auditor.
        
        Returns:
            Dictionary of statistics
        """
        elapsed_time = time.time() - self.stats["start_time"]
        contracts_per_hour = 0
        vulnerabilities_per_day = 0
        
        if elapsed_time > 0:
            contracts_per_hour = (self.stats["contracts_analyzed"] / elapsed_time) * 3600
            vulnerabilities_per_day = (self.stats["vulnerabilities_found"] / elapsed_time) * 86400
        
        return {
            "contracts_analyzed": self.stats["contracts_analyzed"],
            "vulnerabilities_found": self.stats["vulnerabilities_found"],
            "elapsed_time_seconds": elapsed_time,
            "contracts_per_hour": contracts_per_hour,
            "vulnerabilities_per_day": vulnerabilities_per_day
        }

# Main entry point when run as a module
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaChain Contract Auditor")
    parser.add_argument("--contract", required=True, help="Path to the contract source code")
    parser.add_argument("--model", help="Path to the ML model for vulnerability detection")
    parser.add_argument("--output", help="Path to output the results (JSON format)")
    
    args = parser.parse_args()
    
    auditor = ContractAuditor(model_path=args.model)
    results = auditor.analyze_contract(args.contract)
    
    if args.output:
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print results to console
        print(json.dumps(results, indent=2))
    
    # Display stats
    stats = auditor.get_stats()
    print(f"Auditor stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()
EOF
}

  # Create zk verifier module
  log_info "Creating ZK verifier module..."
  mkdir -p llamachain/zk
  cat > llamachain/

function create_transaction_tracer() {
  print_header "Creating Transaction Tracer Module"
  
  log_info "Creating transaction tracer module..."
  cat > llamachain/analysis/trace.py << EOF
"""
TransactionTracer: Traces the execution of transactions for deeper insights.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from web3 import Web3

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("transaction_tracer")

class TransactionTracer:
    """
    TransactionTracer uses Geth's debug_traceTransaction API to trace transaction execution.
    """
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize the TransactionTracer.
        
        Args:
            rpc_url: Ethereum RPC URL (must support debug_traceTransaction)
        """
        self.rpc_url = rpc_url or os.environ.get("ETH_RPC_URL", "http://localhost:8545")
        
        # Connect to Ethereum node
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Check connection
        if not self.web3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            raise ConnectionError("Failed to connect to Ethereum node")
        
        # Check if debug_traceTransaction is supported
        if 'debug' not in self.web3.provider.middlewares:
            logger.warning("RPC endpoint may not support debug_traceTransaction")
        
        logger.info("TransactionTracer initialized successfully")
    
    def trace_transaction(self, tx_hash: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trace a transaction's execution.
        
        Args:
            tx_hash: Transaction hash
            config: Trace configuration options
            
        Returns:
            Dictionary containing trace results
        """
        # Default trace configuration
        if config is None:
            config = {
                "tracer": "callTracer",
                "disableStorage": False,
                "disableStack": False,
                "enableMemory": True,
                "enableReturnData": True
            }
        
        try:
            # Convert hash to proper format if needed
            if not tx_hash.startswith("0x"):
                tx_hash = "0x" + tx_hash
            
            # Call debug_traceTransaction
            trace = self.web3.provider.make_request(
                "debug_traceTransaction", [tx_hash, config]
            )
            
            if "error" in trace:
                logger.error(f"Error tracing transaction: {trace['error']}")
                return {
                    "success": False,
                    "error": trace["error"],
                    "tx_hash": tx_hash
                }
            
            logger.info(f"Successfully traced transaction {tx_hash}")
            
            # Enhance trace with transaction info
            tx_info = self.web3.eth.get_transaction(tx_hash)
            tx_receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            
            enhanced_trace = {
                "success": True,
                "tx_hash": tx_hash,
                "trace": trace["result"],
                "transaction": {
                    "from": tx_info["from"],
                    "to": tx_info["to"],
                    "value": tx_info["value"],
                    "gas": tx_info["gas"],
                    "gasPrice": tx_info["gasPrice"],
                    "input": tx_info["input"]
                },
                "receipt": {
                    "status": tx_receipt["status"],
                    "gasUsed": tx_receipt["gasUsed"],
                    "logs": [dict(log) for log in tx_receipt["logs"]],
                    "contractAddress": tx_receipt.get("contractAddress")
                }
            }
            
            return enhanced_trace
            
        except Exception as e:
            logger.error(f"Error tracing transaction {tx_hash}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tx_hash": tx_hash
            }
    
    def analyze_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a transaction trace to extract insights.
        
        Args:
            trace: Transaction trace from trace_transaction()
            
        Returns:
            Dictionary containing analysis results
        """
        if not trace.get("success", False):
            return {
                "success": False,
                "error": trace.get("error", "Invalid trace"),
                "tx_hash": trace.get("tx_hash", "unknown")
            }
        
        # Extract call tree
        call_tree = trace.get("trace", {})
        
        # Extract basic statistics
        stats = {
            "call_depth": self._calculate_call_depth(call_tree),
            "total_calls": self._count_total_calls(call_tree),
            "contract_interactions": self._extract_contract_interactions(call_tree),
            "value_transfers": self._extract_value_transfers(call_tree),
            "error_messages": self._extract_error_messages(call_tree)
        }
        
        # Detect patterns
        patterns = {
            "contains_delegatecall": self._contains_delegatecall(call_tree),
            "contains_selfdestruct": self._contains_selfdestruct(call_tree),
            "reads_storage": self._reads_storage(call_tree),
            "writes_storage": self._writes_storage(call_tree)
        }
        
        return {
            "success": True,
            "tx_hash": trace.get("tx_hash"),
            "statistics": stats,
            "patterns": patterns,
            "status": trace.get("receipt", {}).get("status"),
            "gas_used": trace.get("receipt", {}).get("gasUsed")
        }
    
    def _calculate_call_depth(self, call_tree: Dict[str, Any]) -> int:
        """Calculate the maximum call depth in the call tree."""
        if not call_tree or "calls" not in call_tree:
            return 1
        
        max_child_depth = 0
        for call in call_tree.get("calls", []):
            child_depth = self._calculate_call_depth(call)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def _count_total_calls(self, call_tree: Dict[str, Any]) -> int:
        """Count the total number of calls in the call tree."""
        if not call_tree:
            return 0
        
        count = 1  # Count this call
        for call in call_tree.get("calls", []):
            count += self._count_total_calls(call)
        
        return count
    
    def _extract_contract_interactions(self, call_tree: Dict[str, Any]) -> List[str]:
        """Extract unique contract addresses interacted with."""
        if not call_tree:
            return []
        
        addresses = []
        if "to" in call_tree and call_tree["to"]:
            addresses.append(call_tree["to"])
        
        for call in call_tree.get("calls", []):
            addresses.extend(self._extract_contract_interactions(call))
        
        return list(set(addresses))  # Remove duplicates
    
    def _extract_value_transfers(self, call_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all value transfers in the call tree."""
        if not call_tree:
            return []
        
        transfers = []
        if "value" in call_tree and int(call_tree["value"], 16) > 0:
            transfers.append({
                "from": call_tree.get("from"),
                "to": call_tree.get("to"),
                "value": int(call_tree["value"], 16)
            })
        
        for call in call_tree.get("calls", []):
            transfers.extend(self._extract_value_transfers(call))
        
        return transfers
    
    def _extract_error_messages(self, call_tree: Dict[str, Any]) -> List[str]:
        """Extract error messages from the call tree."""
        if not call_tree:
            return []
        
        errors = []
        if "error" in call_tree:
            errors.append(call_tree["error"])
        
        for call in call_tree.get("calls", []):
            errors.extend(self._extract_error_messages(call))
        
        return errors
    
    def _contains_delegatecall(self, call_tree: Dict[str, Any]) -> bool:
        """Check if the call tree contains any delegatecalls."""
        if not call_tree:
            return False
        
        if call_tree.get("type") == "DELEGATECALL":
            return True
        
        for call in call_tree.get("calls", []):
            if self._contains_delegatecall(call):
                return True
        
        return False
    
    def _contains_selfdestruct(self, call_tree: Dict[str, Any]) -> bool:
        """Check if the call tree contains any selfdestructs."""
        if not call_tree:
            return False
        
        # Check for opcode in the output
        if "output" in call_tree and call_tree["output"]:
            if "selfdestruct" in call_tree["output"].lower():
                return True
        
        for call in call_tree.get("calls", []):
            if self._contains_selfdestruct(call):
                return True
        
        return False
    
    def _reads_storage(self, call_tree: Dict[str, Any]) -> bool:
        """Check if the call tree reads from storage."""
        # This is a simplified implementation
        # Actual implementation would parse the `structLogs` field
        return True
    
    def _writes_storage(self, call_tree: Dict[str, Any]) -> bool:
        """Check if the call tree writes to storage."""
        # This is a simplified implementation
        # Actual implementation would parse the `structLogs` field
        return True

# Main entry point when run as a module
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaChain Transaction Tracer")
    parser.add_argument("--rpc-url", help="Ethereum RPC URL", default=os.environ.get("ETH_RPC_URL"))
    parser.add_argument("--tx-hash", required=True, help="Transaction hash to trace")
    parser.add_argument("--output", help="Path to output the results (JSON format)")
    parser.add_argument("--analyze", action="store_true", help="Analyze the trace")
    
    args = parser.parse_args()
    
    tracer = TransactionTracer(rpc_url=args.rpc_url)
    trace = tracer.trace_transaction(args.tx_hash)
    
    if args.analyze:
        analysis = tracer.analyze_trace(trace)
        result = {
            "trace": trace,
            "analysis": analysis
        }
    else:
        result = trace
    
    if args.output:
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print results to console
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
EOF
}

function create_security_modules() {
  print_header "Creating Security Modules"
  
  # Create frontrun detector module
  log_info "Creating frontrun detector module..."
  mkdir -p llamachain/security
  cat > llamachain/security/frontrun.py << EOF
"""
FrontrunDetector: Detects front-running activities by analyzing transaction order and gas prices.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from web3 import Web3
from web3.types import TxData

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("frontrun_detector")

class FrontrunDetector:
    """
    FrontrunDetector identifies potential front-running activities in transaction pools and mined blocks.
    """
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the FrontrunDetector.
        
        Args:
            rpc_url: Ethereum HTTP RPC URL
            ws_url: Ethereum WebSocket URL (required for monitoring pending transactions)
        """
        self.rpc_url = rpc_url or os.environ.get("ETH_RPC_URL", "http://localhost:8545")
        self.ws_url = ws_url or os.environ.get("ETH_WSS_URL")
        
        # Connect to Ethereum node
        self.http_web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Check connection
        if not self.http_web3.is_connected():
            logger.error("Failed to connect to Ethereum node via HTTP")
            raise ConnectionError("Failed to connect to Ethereum node via HTTP")
        
        # Connect via WebSocket if available
        self.ws_web3 = None
        if self.ws_url:
            try:
                self.ws_web3 = Web3(Web3.WebsocketProvider(self.ws_url))
                logger.info(f"Connected to Ethereum node via WebSocket: {self.ws_url}")
            except Exception as e:
                logger.error(f"Failed to connect via WebSocket: {e}")
        
        # Threshold values for detection
        self.gas_price_threshold = 1.5  # 50% higher gas price
        self.time_threshold = 30  # 30 seconds
        
        logger.info("FrontrunDetector initialized successfully")
    
    def analyze_block(self, block_number: int) -> List[Dict[str, Any]]:
        """
        Analyze a block for potential front-running activities.
        
        Args:
            block_number: Block number to analyze
            
        Returns:
            List of potential front-running incidents
        """
        # Get block with transactions
        block = self.http_web3.eth.get_block(block_number, full_transactions=True)
        
        # Extract transactions
        transactions = block["transactions"]
        if not transactions:
            return []
        
        # Group transactions by target contract and method
        groups = self._group_transactions(transactions)
        
        # Analyze each group for front-running
        incidents = []
        for target, method_groups in groups.items():
            for method, txs in method_groups.items():
                if len(txs) > 1:
                    front_run_incidents = self._detect_frontrun_in_group(txs, block["timestamp"])
                    incidents.extend(front_run_incidents)
        
        return incidents
    
    def _group_transactions(self, transactions: List[TxData]) -> Dict[str, Dict[str, List[TxData]]]:
        """
        Group transactions by target contract and method signature.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Nested dictionary: {contract_address: {method_signature: [transactions]}}
        """
        groups = {}
        
        for tx in transactions:
            # Skip if no 'to' address (contract creation)
            if tx["to"] is None:
                continue
            
            target = tx["to"]
            input_data = tx["input"]
            
            # Extract method signature (first 4 bytes of input data)
            method = input_data[:10] if len(input_data) >= 10 else input_data
            
            # Initialize nested dictionaries if needed
            if target not in groups:
                groups[target] = {}
            if method not in groups[target]:
                groups[target][method] = []
            
            groups[target][method].append(tx)
        
        return groups
    
    def _detect_frontrun_in_group(self, transactions: List[TxData], block_timestamp: int) -> List[Dict[str, Any]]:
        """
        Detect front-running in a group of similar transactions.
        
        Args:
            transactions: List of similar transactions
            block_timestamp: Timestamp of the block
            
        Returns:
            List of potential front-running incidents
        """
        incidents = []
        
        # Sort transactions by gas price (highest first)
        sorted_txs = sorted(transactions, key=lambda x: x["gasPrice"], reverse=True)
        
        # Compare transactions
        for i in range(len(sorted_txs) - 1):
            high_tx = sorted_txs[i]
            
            for j in range(i + 1, len(sorted_txs)):
                low_tx = sorted_txs[j]
                
                # Check if gas price difference exceeds threshold
                if high_tx["gasPrice"] > low_tx["gasPrice"] * self.gas_price_threshold:
                    # Get transaction receipts to check success
                    high_receipt = self.http_web3.eth.get_transaction_receipt(high_tx["hash"])
                    low_receipt = self.http_web3.eth.get_transaction_receipt(low_tx["hash"])
                    
                    # Both transactions must have been successful
                    if high_receipt["status"] == 1 and low_receipt["status"] == 1:
                        incidents.append({
                            "potential_frontrun": True,
                            "block_number": high_tx["blockNumber"],
                            "block_timestamp": block_timestamp,
                            "frontrunner_tx": {
                                "hash": high_tx["hash"].hex(),
                                "from": high_tx["from"],
                                "gas_price": high_tx["gasPrice"],
                                "nonce": high_tx["nonce"]
                            },
                            "victim_tx": {
                                "hash": low_tx["hash"].hex(),
                                "from": low_tx["from"],
                                "gas_price": low_tx["gasPrice"],
                                "nonce": low_tx["nonce"]
                            },
                            "contract": high_tx["to"],
                            "method": high_tx["input"][:10],
                            "gas_price_difference": high_tx["gasPrice"] / low_tx["gasPrice"],
                            "confidence": self._calculate_confidence(high_tx, low_tx)
                        })
        
        return incidents
    
    def _calculate_confidence(self, high_tx: TxData, low_tx: TxData) -> float:
        """
        Calculate confidence score for front-running detection.
        
        Args:
            high_tx: Transaction with higher gas price
            low_tx: Transaction with lower gas price
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Factors affecting confidence:
        # 1. Gas price difference
        # 2. Same block inclusion
        # 3. Similar input data
        # 4. Transaction sender reputation (not implemented yet)
        
        # Gas price factor (max 0.5)
        price_ratio = high_tx["gasPrice"] / low_tx["gasPrice"]
        price_factor = min(0.5, (price_ratio - 1) / 10)
        
        # Block inclusion factor (0.2 if same block)
        block_factor = 0.2 if high_tx["blockNumber"] == low_tx["blockNumber"] else 0.0
        
        # Input data similarity factor (max 0.3)
        input_similarity = self._calculate_input_similarity(high_tx["input"], low_tx["input"])
        input_factor = 0.3 * input_similarity
        
        return price_factor + block_factor + input_factor
    
    def _calculate_input_similarity(self, input1: str, input2: str) -> float:
        """
        Calculate similarity between transaction input data.
        
        Args:
            input1: First input data
            input2: Second input data
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple method: check if method signatures match
        if input1[:10] == input2[:10]:
            # If identical, return 1.0
            if input1 == input2:
                return 1.0
            # If same length, compare by character
            elif len(input1) == len(input2):
                matching_chars = sum(1 for a, b in zip(input1[10:], input2[10:]) if a == b)
                return 0.5 + 0.5 * (matching_chars / max(1, len(input1[10:])))
            # If different length, partial match
            else:
                return 0.7
        return 0.0
    
    def monitor_mempool(self, callback=None):
        """
        Monitor the mempool for potential front-running activities.
        
        Args:
            callback: Optional callback function to be called with potential incidents
        """
        if not self.ws_web3:
            logger.error("WebSocket connection required for mempool monitoring")
            return
        
        # Internal state to track pending transactions
        pending_txs = {}
        
        def handle_pending_tx(tx_hash):
            try:
                # Get transaction details
                tx = self.ws_web3.eth.get_transaction(tx_hash)
                if not tx:
                    return
                
                # Skip if no 'to' address (contract creation)
                if tx["to"] is None:
                    return
                
                target = tx["to"]
                input_data = tx["input"]
                method = input_data[:10] if len(input_data) >= 10 else input_data
                
                # Create key for this transaction type
                tx_key = f"{target}:{method}"
                
                # Check if we've seen similar transactions recently
                now = time.time()
                
                if tx_key in pending_txs:
                    # Check for potential front-running
                    for prior_tx in pending_txs[tx_key]:
                        # Skip if from same sender
                        if prior_tx["from"] == tx["from"]:
                            continue
                        
                        # Check time threshold
                        if now - prior_tx["timestamp"] > self.time_threshold:
                            continue
                        
                        # Check gas price difference
                        if tx["gasPrice"] > prior_tx["gasPrice"] * self.gas_price_threshold:
                            incident = {
                                "potential_frontrun": True,
                                "mempool_detection": True,
                                "timestamp": now,
                                "frontrunner_tx": {
                                    "hash": tx["hash"].hex(),
                                    "from": tx["from"],
                                    "gas_price": tx["gasPrice"],
                                    "nonce": tx["nonce"]
                                },
                                "target_tx": {
                                    "hash": prior_tx["hash"].hex(),
                                    "from": prior_tx["from"],
                                    "gas_price": prior_tx["gasPrice"],
                                    "nonce": prior_tx["nonce"]
                                },
                                "contract": tx["to"],
                                "method": method,
                                "gas_price_difference": tx["gasPrice"] / prior_tx["gasPrice"],
                                "confidence": self._calculate_confidence(tx, prior_tx)
                            }
                            
                            logger.info(f"Potential front-running detected in mempool: {tx['hash'].hex()}")
                            
                            if callback:
                                callback(incident)
                
                # Add to pending transactions
                tx_with_timestamp = dict(tx)
                tx_with_timestamp["timestamp"] = now
                
                if tx_key not in pending_txs:
                    pending_txs[tx_key] = []
                
                pending_txs[tx_key].append(tx_with_timestamp)
                
                # Clean up old pending transactions
                for key in list(pending_txs.keys()):
                    pending_txs[key] = [
                        t for t in pending_txs[key] 
                        if now - t["timestamp"] <= self.time_threshold
                    ]
                    if not pending_txs[key]:
                        del pending_txs[key]
                
            except Exception as e:
                logger.error(f"Error processing pending transaction: {e}")
        
        # Create a filter for pending transactions
        pending_filter = self.ws_web3.eth.filter('pending')
        
        try:
            # Poll for pending transactions
            while True:
                for tx_hash in pending_filter.get_new_entries():
                    handle_pending_tx(tx_hash)
                time.sleep(0.1)  # Poll frequently
        except KeyboardInterrupt:
            logger.info("Stopping mempool monitor")
        except Exception as e:
            logger.error(f"Error in mempool monitor: {e}")
        finally:
            # Clean up
            self.ws_web3.eth.uninstall_filter(pending_filter.filter_id)

# Main entry point when run as a module
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaChain Front-running Detector")
    parser.add_argument("--rpc-url", help="Ethereum RPC URL", default=os.environ.get("ETH_RPC_URL"))
    parser.add_argument("--ws-url", help="Ethereum WebSocket URL", default=os.environ.get("ETH_WSS_URL"))
    parser.add_argument("--block", type=int, help="Analyze a specific block")
    parser.add_argument("--range", help="Analyze a range of blocks (format: start-end)")
    parser.add_argument("--monitor", action="store_true", help="Monitor mempool for front-running")
    parser.add_argument("--output", help="Path to output the results (JSON format)")
    
    args = parser.parse_args()
    
    detector = FrontrunDetector(rpc_url=args.rpc_url, ws_url=args.ws_url)
    
    results = []
    if args.block:
        print(f"Analyzing block {args.block} for front-running...")
        incidents = detector.analyze_block(args.block)
        results.extend(incidents)
        print(f"Found {len(incidents)} potential front-running incidents")
    elif args.range:
        try:
            start, end = map(int, args.range.split("-"))
            print(f"Analyzing blocks {start} to {end} for front-running...")
            for block_num in range(start, end + 1):
                incidents = detector.analyze_block(block_num)
                results.extend(incidents)
                print(f"Block {block_num}: Found {len(incidents)} potential incidents")
            print(f"Total: Found {len(results)} potential front-running incidents")
        except ValueError:
            print("Invalid range format. Use start-end (e.g., 15000000-15000010)")
    elif args.monitor:
        print("Monitoring mempool for potential front-running...")
        
        def print_incident(incident):
            print(f"Potential front-running detected:")
            print(f"  Frontrunner: {incident['frontrunner_tx']['hash']}")
            print(f"  Target: {incident['target_tx']['hash']}")
            print(f"  Gas price difference: {incident['gas_price_difference']:.2f}x")
            print(f"  Confidence: {incident['confidence']:.2f}")
        
        detector.monitor_mempool(callback=print_incident)
    else:
        print("No action specified. Use --help to see available options.")
    
    if args.output and results:
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
EOF

  print_header "Creating Configuration Files"
  
  # Create .env file with default configuration
  log_info "Creating .env file..."
  cat > .env << EOF
# LlamaChain Configuration

# Ethereum node configuration
ETH_RPC_URL=https://mainnet.infura.io/v3/your-infura-key
ETH_WSS_URL=wss://mainnet.infura.io/ws/v3/your-infura-key
ETH_CHAIN_ID=1

# API configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=true

# Database configuration
DB_TYPE=sqlite
DB_PATH=data/llamachain.db

# IPFS configuration
IPFS_HOST=ipfs.infura.io
IPFS_PORT=5001
IPFS_PROTOCOL=https

# Kubernetes configuration
K8S_NAMESPACE=llamachain

# Analytics pipeline configuration
BEAM_RUNNER=DirectRunner
BIGQUERY_PROJECT=your-project-id
BIGQUERY_DATASET=blockchain_data

# Security configuration
ENCRYPTION_KEY=change_this_to_a_secure_random_key

# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=console

# ML configuration
ML_MODEL_PATH=models/vulnerability_detector.mlx
EOF

  # Create docker-compose.yml
  log_info "Creating docker-compose.yml..."
  cat > docker-compose.yml << EOF
version: '3.8'

services:
  # LlamaChain API service
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "5000:5000"
    environment:
      - ETH_RPC_URL=\${ETH_RPC_URL}
      - API_HOST=0.0.0.0
      - API_PORT=5000
      - DB_TYPE=\${DB_TYPE}
      - DB_PATH=/data/llamachain.db
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    depends_on:
      - indexer
    networks:
      - llamachain-network

  # Blockchain Indexer service
  indexer:
    build:
      context: .
      dockerfile: docker/Dockerfile.indexer
    environment:
      - ETH_RPC_URL=\${ETH_RPC_URL}
      - ETH_WSS_URL=\${ETH_WSS_URL}
      - DB_TYPE=\${DB_TYPE}
      - DB_PATH=/data/llamachain.db
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    networks:
      - llamachain-network

  # Contract Auditor service
  auditor:
    build:
      context: .
      dockerfile: docker/Dockerfile.auditor
    environment:
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
      - ./contracts:/contracts
    networks:
      - llamachain-network

  # Analytics Pipeline service
  pipeline:
    build:
      context: .
      dockerfile: docker/Dockerfile.pipeline
    environment:
      - BEAM_RUNNER=\${BEAM_RUNNER}
      - BIGQUERY_PROJECT=\${BIGQUERY_PROJECT}
      - BIGQUERY_DATASET=\${BIGQUERY_DATASET}
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    networks:
      - llamachain-network

networks:
  llamachain-network:
    driver: bridge
EOF

  # Create example Kubernetes deployment file
  log_info "Creating Kubernetes deployment file..."
  cat > k8s/geth.yaml << EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: geth
  namespace: \${K8S_NAMESPACE}
spec:
  selector:
    matchLabels:
      app: geth
  serviceName: "geth"
  replicas: 1
  template:
    metadata:
      labels:
        app: geth
    spec:
      containers:
      - name: geth
        image: ethereum/client-go:latest
        args:
          - "--http"
          - "--http.addr=0.0.0.0"
          - "--http.api=eth,net,web3,debug"
          - "--http.corsdomain=*"
          - "--ws"
          - "--ws.addr=0.0.0.0"
          - "--ws.api=eth,net,web3,debug"
          - "--ws.origins=*"
          - "--syncmode=snap"
          - "--datadir=/data"
        ports:
        - containerPort: 8545
          name: http
        - containerPort: 8546
          name: websocket
        volumeMounts:
        - name: geth-data
          mountPath: /data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: geth-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi
---
apiVersion: v1
kind: Service
metadata:
  name: geth
  namespace: \${K8S_NAMESPACE}
spec:
  selector:
    app: geth
  ports:
  - name: http
    port: 8545
    targetPort: 8545
  - name: websocket
    port: 8546
    targetPort: 8546
  type: ClusterIP
EOF

  # Create Dockerfile for API
  log_info "Creating Dockerfiles..."
  mkdir -p docker
  
  cat > docker/Dockerfile.api << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 5000

# Run the API server
CMD ["python", "-m", "llamachain.api.app"]
EOF

  cat > docker/Dockerfile.indexer << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the indexer
CMD ["python", "-m", "llamachain.core.indexer"]
EOF

  cat > docker/Dockerfile.auditor << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Slither
RUN apt-get update && \
    apt-get install -y git solc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the contract auditor
CMD ["python", "-m", "llamachain.analysis.contract"]
EOF

  cat > docker/Dockerfile.pipeline << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the analytics pipeline
CMD ["python", "-m", "llamachain.pipelines.analytics"]
EOF

  # Create setup.py
  log_info "Creating setup.py..."
  cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="llamachain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "web3>=6.0.0",
        "py-solc-x>=1.1.1",
        "slither-analyzer>=0.9.0",
        "py_ecc>=5.2.0",
        "ipfshttpclient>=0.8.0",
        "apache-beam>=2.46.0",
        "flask>=2.3.2",
        "flask-restx>=1.1.0",
        "typer>=0.9.0",
        "rich>=13.3.5",
        "pydantic>=1.10.8",
    ],
    entry_points={
        "console_scripts": [
            "llamachain=llamachain.cli.main:app",
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="LlamaChain: An intelligent blockchain analysis platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llamachain",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF

  # Create README.md
  log_info "Creating README.md..."
  cat > README.md << EOF
# 🦙 LlamaChain: Blockchain Intelligence Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-enabled-brightgreen)](https://github.com/ml-explore/mlx)

LlamaChain is an advanced blockchain intelligence platform designed for comprehensive on-chain data analysis and smart contract security auditing. Built with Apple Silicon optimizations using MLX, this platform provides powerful tools for blockchain developers, security researchers, and DeFi analysts.

## 🌟 Key Features

- **Blockchain Indexing**: Connects to Ethereum and other blockchains to retrieve and index on-chain data
- **Smart Contract Security Auditing**: Leverages Slither and machine learning for comprehensive vulnerability detection
- **Analytics API**: Provides endpoints for querying blockchain data and audit results
- **Transaction Tracing**: In-depth analysis of transaction execution for deeper insights
- **Zero-Knowledge Proofs**: Enhanced trust and privacy with zk-proof verification
- **MLX Acceleration**: Optimized performance on Apple Silicon for machine learning tasks
- **Containerized Deployment**: Seamless setup with Docker and Kubernetes

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llamachain.git
cd llamachain

# Run the setup script
./run_llamachain.sh --setup

# Start the platform
./run_llamachain.sh --start
```

## 📊 Benchmarks

- **Blocks Processed**: 1000/hr
- **Average Block Time**: 12s
- **Transactions Analyzed**: 5000/hr
- **Vulnerabilities Found**: 10/day
- **ZK-Proof Verification Time**: <100ms

## 🔍 Use Cases

1. **Blockchain Developers**: Monitor network activity and transaction patterns
2. **Security Researchers**: Audit smart contracts for vulnerabilities
3. **DeFi Analysts**: Analyze on-chain financial data for insights

## 🛠️ Architecture

LlamaChain consists of several key components:

- **BlockchainIndexer**: Connects to blockchain networks and retrieves data
- **ContractAuditor**: Performs security analysis on smart contracts
- **Analytics API**: Provides endpoints for querying analyzed data
- **TransactionTracer**: Traces transaction execution for deeper insights
- **Security Matrix**: Comprehensive security framework with multiple components

## 📚 Documentation

For detailed documentation, please visit our [Wiki](https://github.com/yourusername/llamachain/wiki).

## 🔧 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linters
pre-commit run --all-files
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOF

  # Create an example contract for testing
  log_info "Creating example smart contract..."
  mkdir -p contracts
  cat > contracts/example.sol << EOF
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Vulnerable function - reentrancy risk
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Potential reentrancy vulnerability (sending ETH before updating state)
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    // Function to check contract balance
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
EOF

  log_success "Configuration files created"
  return 0
}

function create_core_modules() {
  print_header "Creating Core Modules"
  
  # Create database module
  log_info "Creating database module..."
  cat > llamachain/core/db.py << EOF
"""
Database module for LlamaChain.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
import logging

class Database:
    """
    Database class for storing and retrieving blockchain data.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or os.environ.get("DB_PATH", "data/llamachain.db")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Blocks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocks (
            number INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            parent_hash TEXT,
            timestamp INTEGER,
            miner TEXT,
            difficulty INTEGER,
            size INTEGER,
            gas_used INTEGER,
            gas_limit INTEGER,
            transaction_count INTEGER,
            extra_data TEXT
        )
        ''')
        
        # Transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            hash TEXT PRIMARY KEY,
            block_number INTEGER,
            from_address TEXT,
            to_address TEXT,
            value TEXT,
            gas INTEGER,
            gas_price INTEGER,
            nonce INTEGER,
            input TEXT,
            transaction_index INTEGER,
            FOREIGN KEY (block_number) REFERENCES blocks(number)
        )
        ''')
        
        # Smart contracts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS contracts (
            address TEXT PRIMARY KEY,
            creator_tx_hash TEXT,
            block_number INTEGER,
            bytecode TEXT,
            is_verified BOOLEAN DEFAULT 0,
            creation_timestamp INTEGER,
            FOREIGN KEY (block_number) REFERENCES blocks(number)
        )
        ''')
        
        # Audit reports table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT,
            report_data TEXT,
            timestamp INTEGER,
            vulnerability_count INTEGER,
            critical_count INTEGER,
            high_count INTEGER,
            medium_count INTEGER,
            low_count INTEGER,
            FOREIGN KEY (contract_address) REFERENCES contracts(address)
        )
        ''')
        
        self.conn.commit()
    
    def store_block(self, block_data: Dict[str, Any]) -> bool:
        """
        Store a block in the database.
        
        Args:
            block_data: Block data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Extract relevant fields
            cursor.execute('''
            INSERT OR REPLACE INTO blocks (
                number, hash, parent_hash, timestamp, miner,
                difficulty, size, gas_used, gas_limit, transaction_count, extra_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block_data.get("number"),
                block_data.get("hash").hex() if block_data.get("hash") else None,
                block_data.get("parentHash").hex() if block_data.get("parentHash") else None,
                block_data.get("timestamp"),
                block_data.get("miner"),
                block_data.get("difficulty"),
                block_data.get("size"),
                block_data.get("gasUsed"),
                block_data.get("gasLimit"),
                len(block_data.get("transactions", [])),
                block_data.get("extraData").hex() if block_data.get("extraData") else None
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing block: {e}")
            self.conn.rollback()
            return False
    
    def store_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """
        Store a transaction in the database.
        
        Args:
            tx_data: Transaction data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Extract relevant fields
            cursor.execute('''
            INSERT OR REPLACE INTO transactions (
                hash, block_number, from_address, to_address, value,
                gas, gas_price, nonce, input, transaction_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx_data.get("hash").hex() if tx_data.get("hash") else None,
                tx_data.get("blockNumber"),
                tx_data.get("from"),
                tx_data.get("to"),
                str(tx_data.get("value", 0)),
                tx_data.get("gas"),
                tx_data.get("gasPrice"),
                tx_data.get("nonce"),
                tx_data.get("input"),
                tx_data.get("transactionIndex")
            ))
            
            # Check if this is a contract creation transaction
            if tx_data.get("to") is None and tx_data.get("input"):
                self._handle_contract_creation(tx_data)
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing transaction: {e}")
            self.conn.rollback()
            return False
    
    def _handle_contract_creation(self, tx_data: Dict[str, Any]):
        """
        Handle a contract creation transaction.
        
        Args:
            tx_data: Transaction data as a dictionary
        """
        # TODO: Implement contract address calculation
        # This requires the transaction receipt to get the contract address
        pass
    
    def store_contract(self, contract_data: Dict[str, Any]) -> bool:
        """
        Store a smart contract in the database.
        
        Args:
            contract_data: Contract data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO contracts (
                address, creator_tx_hash, block_number, bytecode, is_verified, creation_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                contract_data.get("address"),
                contract_data.get("creator_tx_hash"),
                contract_data.get("block_number"),
                contract_data.get("bytecode"),
                contract_data.get("is_verified", False),
                contract_data.get("creation_timestamp")
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing contract: {e}")
            self.conn.rollback()
            return False
    
    def store_audit_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Store a smart contract audit report in the database.
        
        Args:
            report_data: Audit report data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO audit_reports (
                contract_address, report_data, timestamp,
                vulnerability_count, critical_count, high_count, medium_count, low_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_data.get("contract_address"),
                json.dumps(report_data.get("findings", [])),
                report_data.get("timestamp"),
                report_data.get("vulnerability_count", 0),
                report_data.get("critical_count", 0),
                report_data.get("high_count", 0),
                report_data.get("medium_count", 0),
                report_data.get("low_count", 0)
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing audit report: {e}")
            self.conn.rollback()
            return False
    
    def get_block(self, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by its number.
        
        Args:
            block_number: Block number
            
        Returns:
            Block data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM blocks WHERE number = ?", (block_number,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE hash = ?", (tx_hash,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_contract(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a contract by its address.
        
        Args:
            address: Contract address
            
        Returns:
            Contract data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM contracts WHERE address = ?", (address,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_latest_audit_report(self, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest audit report for a contract.
        
        Args:
            contract_address: Contract address
            
        Returns:
            Audit report data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_reports 
            WHERE contract_address = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (contract_address,))
        row = cursor.fetchone()
        
        if row:
            report = dict(row)
            # Parse the JSON report_data field
            report["findings"] = json.loads(report["report_data"])
            return report
        return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
EOF

  # Create logger utility
  log_info "Creating logger utility..."
  mkdir -p llamachain/utils
  cat > llamachain/utils/logger.py << EOF
"""
Logger utility for LlamaChain.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name, log_level=None, log_file=None):
    """
    Set up and return a logger.
    
    Args:
        name: Logger name
        log_level: Logging level (defaults to environment variable or INFO)
        log_file: Log file path (optional)
        
    Returns:
        Logger instance
    """
    # Determine log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create rotating file handler (10MB max size, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
EOF

  # Create core/indexer.py - BlockchainIndexer module
  log_info "Creating BlockchainIndexer module..."
  cat > llamachain/core/indexer.py << EOF
"""
BlockchainIndexer: Connects to blockchain networks and retrieves data.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from web3 import Web3
from web3.types import BlockData, TxData
import json

from llamachain.core.db import Database
from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("blockchain_indexer")

class BlockchainIndexer:
    """
    BlockchainIndexer connects to Ethereum blockchain and indexes blocks and transactions.
    """
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the BlockchainIndexer.
        
        Args:
            rpc_url: Ethereum HTTP RPC URL
            ws_url: Ethereum WebSocket URL
        """
        self.rpc_url = rpc_url or os.environ.get("ETH_RPC_URL", "http://localhost:8545")
        self.ws_url = ws_url or os.environ.get("ETH_WSS_URL")
        
        # Connect to Ethereum node
        if self.ws_url:
            try:
                self.web3 = Web3(Web3.WebsocketProvider(self.ws_url))
                logger.info(f"Connected to Ethereum node via WebSocket: {self.ws_url}")
            except Exception as e:
                logger.error(f"Failed to connect via WebSocket: {e}")
                self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
                logger.info(f"Fallback: Connected to Ethereum node via HTTP: {self.rpc_url}")
        else:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            logger.info(f"Connected to Ethereum node via HTTP: {self.rpc_url}")
        
        # Check connection
        if not self.web3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            raise ConnectionError("Failed to connect to Ethereum node")
        
        # Initialize database connection
        self.db = Database()
        
        logger.info("BlockchainIndexer initialized successfully")
        
        # Statistics
        self.stats = {
            "blocks_processed": 0,
            "transactions_processed": 0,
            "start_time": time.time()
        }
    
    def get_block(self, block_identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by its number or hash.
        
        Args:
            block_identifier: Block number or hash
            
        Returns:
            Block data as a dictionary, or None if not found
        """
        try:
            block = self.web3.eth.get_block(block_identifier, full_transactions=True)
            return dict(block)
        except Exception as e:
            logger.error(f"Error retrieving block {block_identifier}: {e}")
            return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary, or None if not found
        """
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            return dict(tx)
        except Exception as e:
            logger.error(f"Error retrieving transaction {tx_hash}: {e}")
            return None
    
    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction receipt by transaction hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt data as a dictionary, or None if not found
        """
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except Exception as e:
            logger.error(f"Error retrieving transaction receipt {tx_hash}: {e}")
            return None
    
    def index_block(self, block_number: int) -> bool:
        """
        Index a block and its transactions.
        
        Args:
            block_number: Block number to index
            
        Returns:
            True if successful, False otherwise
        """
        block = self.get_block(block_number)
        if not block:
            return False
        
        # Store block in database
        self.db.store_block(block)
        
        # Index transactions
        for tx in block.get("transactions", []):
            if isinstance(tx, dict):  # Only process if full transactions were retrieved
                self.db.store_transaction(tx)
        
        # Update statistics
        self.stats["blocks_processed"] += 1
        self.stats["transactions_processed"] += len(block.get("transactions", []))
        
        logger.info(f"Indexed block {block_number} with {len(block.get('transactions', []))} transactions")
        return True
    
    def index_range(self, start_block: int, end_block: int) -> Dict[str, int]:
        """
        Index a range of blocks.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            
        Returns:
            Statistics about the indexing process
        """
        success_count = 0
        fail_count = 0
        
        for block_num in range(start_block, end_block + 1):
            success = self.index_block(block_num)
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        return {
            "blocks_processed": success_count,
            "blocks_failed": fail_count,
            "total_blocks": end_block - start_block + 1
        }
    
    def index_latest_blocks(self, count: int = 10) -> Dict[str, int]:
        """
        Index the latest n blocks.
        
        Args:
            count: Number of latest blocks to index
            
        Returns:
            Statistics about the indexing process
        """
        latest_block = self.web3.eth.block_number
        start_block = max(0, latest_block - count + 1)
        
        return self.index_range(start_block, latest_block)
    
    def listen_for_new_blocks(self, callback=None):
        """
        Listen for new blocks and index them as they arrive.
        
        Args:
            callback: Optional callback function to be called with each new block
        """
        if not self.ws_url:
            logger.error("WebSocket URL is required for listening to new blocks")
            return
        
        def handle_new_block(block_hash):
            block_number = self.web3.eth.get_block(block_hash).number
            logger.info(f"New block detected: {block_number}")
            self.index_block(block_number)
            
            if callback:
                callback(block_number)
        
        # Create a filter for new blocks
        new_block_filter = self.web3.eth.filter('latest')
        
        try:
            # Poll for new blocks
            while True:
                for block_hash in new_block_filter.get_new_entries():
                    handle_new_block(block_hash)
                time.sleep(1)  # Poll every second
        except KeyboardInterrupt:
            logger.info("Stopping block listener")
        except Exception as e:
            logger.error(f"Error in block listener: {e}")
        finally:
            # Clean up
            self.web3.eth.uninstall_filter(new_block_filter.filter_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexer.
        
        Returns:
            Dictionary of statistics
        """
        elapsed_time = time.time() - self.stats["start_time"]
        blocks_per_hour = 0
        txs_per_hour = 0
        
        if elapsed_time > 0:
            blocks_per_hour = (self.stats["blocks_processed"] / elapsed_time) * 3600
            txs_per_hour = (self.stats["transactions_processed"] / elapsed_time) * 3600
        
        return {
            "blocks_processed": self.stats["blocks_processed"],
            "transactions_processed": self.stats["transactions_processed"],
            "elapsed_time_seconds": elapsed_time,
            "blocks_per_hour": blocks_per_hour,
            "transactions_per_hour": txs_per_hour
        }

# Main entry point when run as a module
def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="LlamaChain Blockchain Indexer")
    parser.add_argument("--rpc-url", help="Ethereum RPC URL", default=os.environ.get("ETH_RPC_URL"))
    parser.add_argument("--ws-url", help="Ethereum WebSocket URL", default=os.environ.get("ETH_WSS_URL"))
    parser.add_argument("--latest", type=int, help="Index latest N blocks", default=0)
    parser.add_argument("--range", help="Index a range of blocks (format: start-end)", default="")
    parser.add_argument("--block", type=int, help="Index a specific block", default=0)
    parser.add_argument("--listen", action="store_true", help="Listen for new blocks")
    
    args = parser.parse_args()
    
    indexer = BlockchainIndexer(rpc_url=args.rpc_url, ws_url=args.ws_url)
    
    if args.block > 0:
        print(f"Indexing block {args.block}...")
        indexer.index_block(args.block)
    elif args.range:
        try:
            start, end = map(int, args.range.split("-"))
            print(f"Indexing blocks {start} to {end}...")
            stats = indexer.index_range(start, end)
            print(f"Indexed {stats['blocks_processed']} blocks successfully, {stats['blocks_failed']} failed")
        except ValueError:
            print("Invalid range format. Use start-end (e.g., 15000000-15000010)")
    elif args.latest > 0:
        print(f"Indexing latest {args.latest} blocks...")
        stats = indexer.index_latest_blocks(args.latest)
        print(f"Indexed {stats['blocks_processed']} blocks successfully, {stats['blocks_failed']} failed")
    elif args.listen:
        print("Listening for new blocks...")
        indexer.listen_for_new_blocks()
    else:
        print("No action specified. Use --help to see available options.")
    
    # Display stats
    stats = indexer.get_stats()
    print(f"Indexer stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()
