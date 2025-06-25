"""Database models and connection management."""

from datetime import datetime
from sqlmodel import Field, Session, SQLModel, create_engine
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Database Models ----------------------------
class Doc(SQLModel, table=True):
    """Document metadata model."""
    id: str = Field(primary_key=True)
    name: str
    status: str = "ready"  # ready | failed
    n_chunks: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

# -------------------- Database Setup and Connection ----------------
def setup_database_directory():
    """Ensure database directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Database directory ensured: {data_dir.absolute()}")
    return data_dir

def get_database_url():
    """Get database URL with proper directory setup."""
    data_dir = setup_database_directory()
    db_path = data_dir / "app.db"
    # Use absolute path for SQLite to avoid path issues
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{db_path.absolute()}")
    logger.info(f"Database URL: {database_url}")
    return database_url

# Initialize database connection
DATABASE_URL = get_database_url()
logger.info(f"Connecting to database: {DATABASE_URL}")
engine = create_engine(DATABASE_URL, echo=False)

def create_tables():
    """Create all database tables."""
    try:
        logger.info("Creating database tables...")
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise e

def get_db_session() -> Session:
    """Get a database session."""
    try:
        return Session(engine)
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        raise e

def init_database():
    """Initialize database - call this when the application starts."""
    logger.info("Initializing database...")
    setup_database_directory()  # Ensure directory exists
    create_tables()
    logger.info("Database initialization completed successfully")

# Don't initialize tables at import time - wait for explicit initialization 