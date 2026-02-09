from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger
from src.configs import settings
from pipelines import telegram_data_etl

@click.command(
    help="""
Telegram Data Agent CLI v0.0.1. 
Main entry point for the Telegram Data ETL pipeline execution.
Run the ZenML Telegram Data Agent project pipelines with various options.
Run a pipeline with the required parameters. This executes
all steps in the pipeline in the correct order using the orchestrator
stack component that is configured in your active ZenML stack.

Examples:

  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache
  
  \b
  # Run only the ETL pipeline
  python run.py --only-etl

"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-sink-to-mongodb",
    is_flag=True,
    default=False,
    help="Whether to run all the data pipelines in one go.",
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
def main(no_cache: bool, 
         run_sink_to_mongodb: bool, 
         run_etl: bool,
         etl_config_filename: str ,
         export_settings: bool = False) -> None:
    assert (run_sink_to_mongodb or run_etl),"At least one pipeline must be selected to run."
    
    if export_settings:
        logger.info("Exporting settings to ZenML secrets.")
        settings.export()
        
    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent
    
    if run_sink_to_mongodb:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "config" / "full_initial_telegram_load.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"full_initial_load_sink_into_mongodb_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        telegram_data_etl.with_options(**pipeline_args)(**run_args_etl)