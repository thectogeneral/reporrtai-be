"""Add google_id column to users table

Revision ID: 20241217_000001
Revises: 
Create Date: 2024-12-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20241217_000001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add google_id column to users table
    op.add_column('users', sa.Column('google_id', sa.String(255), nullable=True))
    
    # Create unique index on google_id
    op.create_index('ix_users_google_id', 'users', ['google_id'], unique=True)


def downgrade() -> None:
    # Remove index first
    op.drop_index('ix_users_google_id', table_name='users')
    
    # Remove google_id column
    op.drop_column('users', 'google_id')

