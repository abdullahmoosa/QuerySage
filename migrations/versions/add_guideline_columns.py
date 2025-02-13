"""add guideline columns

Revision ID: 1a2b3c4d5e6f
Create Date: 2024-03-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = 'a1b2c3d4e5f6'  # Points to the initial schema migration
branch_labels = None
depends_on = None


def upgrade():
    # Add guideline_text_box first (nullable by default)
    op.add_column('questions', sa.Column('guideline_text_box', sa.Text(), nullable=True))
    
    # Add is_textbox with default false for new records
    op.add_column('questions', sa.Column('is_textbox', sa.Boolean(), nullable=False, server_default='false'))
    
    # Add sample_answers_textbox column
    op.add_column('questions', sa.Column('sample_answers_textbox', sa.Text(), nullable=True))
    
    # Add sample_answers_vector_db_url column
    op.add_column('questions', sa.Column('sample_answers_vector_db_url', sa.Text(), nullable=True))


def downgrade():
    # Remove the new columns
    op.drop_column('questions', 'sample_answers_vector_db_url')
    op.drop_column('questions', 'sample_answers_textbox')
    op.drop_column('questions', 'is_textbox')
    op.drop_column('questions', 'guideline_text_box') 