"""remove sample_answers array

Revision ID: ac1258e6b70a
Create Date: 2024-03-20 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ac1258e6b70a'
down_revision = '1a2b3c4d5e6f'  # Points to the guideline columns migration
branch_labels = None
depends_on = None


def upgrade():
    # Remove the sample_answers array column
    op.drop_column('questions', 'sample_answers')


def downgrade():
    # Add back the sample_answers array column if needed to rollback
    op.add_column('questions', sa.Column('sample_answers', sa.ARRAY(sa.Text()), nullable=True))
