"""Database models and connection management."""

from datetime import datetime
from sqlmodel import Field, Session, SQLModel, create_engine
from src.core.config import DATABASE_URL

# -------------------- Database Models ----------------------------
class Doc(SQLModel, table=True):
    """Document metadata model."""
    id: str = Field(primary_key=True)
    name: str
    status: str = "ready"  # ready | failed
    n_chunks: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

# -------------------- Database Connection ------------------------
engine = create_engine(DATABASE_URL, echo=False)

def create_tables():
    """Create all database tables."""
    SQLModel.metadata.create_all(engine)

def get_db_session() -> Session:
    """Get a database session."""
    return Session(engine)

# Initialize tables
create_tables() 