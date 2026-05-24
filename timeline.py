import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def inspect_events() -> None:
    """Inspect events.tsv file structure and contents."""
    data_dir = Path("./data")
    event_files = list(data_dir.rglob("*_events.tsv"))
    
    if not event_files:
        logger.warning("Could not find an events.tsv file")
        return
        
    tsv_path = event_files[0]
    logger.info(f"Found timeline file at: {tsv_path}")
    
    df = pd.read_csv(tsv_path, sep='\t')
    
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"First 15 events:\n{df.head(15)}")
    
    if 'value' in df.columns:
        logger.info(f"Event trigger counts:\n{df['value'].value_counts()}")
    elif 'trial_type' in df.columns:
        logger.info(f"Event type counts:\n{df['trial_type'].value_counts()}")

if __name__ == "__main__":
    inspect_events()
