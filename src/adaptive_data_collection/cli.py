"""
Command-line interface for the Adaptive Data Collection System.
"""

import click
import sys
from pathlib import Path
from .config import CollectionConfig


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Adaptive Data Collection System CLI."""
    pass


@cli.command()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
def collect(config, dry_run):
    """Start data collection process."""
    try:
        # Load configuration
        if config:
            collection_config = CollectionConfig.from_yaml(config)
        else:
            collection_config = CollectionConfig.from_env()
        
        # Validate configuration
        collection_config.validate()
        
        if dry_run:
            click.echo("Configuration loaded successfully:")
            click.echo(f"  API Key: {'Set' if collection_config.alpha_vantage_api_key else 'Not Set'}")
            click.echo(f"  Rate Limit: {collection_config.alpha_vantage_rpm} RPM")
            click.echo(f"  Years to Collect: {collection_config.years_to_collect}")
            click.echo(f"  Output Directory: {collection_config.output_base_dir}")
            click.echo(f"  Symbols File: {collection_config.us_symbols_file}")
            return
        
        click.echo("Starting data collection...")
        # TODO: Implement actual collection logic
        click.echo("Collection logic not yet implemented.")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def validate():
    """Validate configuration and system requirements."""
    try:
        config = CollectionConfig.from_env()
        config.validate()
        
        click.echo("✓ Configuration is valid")
        click.echo(f"✓ API Key: {'Set' if config.alpha_vantage_api_key else 'Not Set'}")
        click.echo(f"✓ Symbols file exists: {Path(config.us_symbols_file).exists()}")
        click.echo(f"✓ Output directory: {config.output_base_dir}")
        
    except Exception as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show collection status and progress."""
    click.echo("Status command not yet implemented.")


if __name__ == "__main__":
    cli()