"""initial schema

Revision ID: a1b2c3d4e5f6
Create Date: 2024-03-20 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create courses table
    op.create_table(
        'courses',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('user_id', UUID, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP'))
    )

    # Create subjects table
    op.create_table(
        'subjects',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.id')),
        sa.Column('vector_db_url', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP'))
    )

    # Create questions table
    op.create_table(
        'questions',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('guideline_vector_db_url', sa.Text),
        sa.Column('sample_answers', sa.ARRAY(sa.Text)),
        sa.Column('instructions', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('subject_id', UUID, sa.ForeignKey('subjects.id'))
    )


def downgrade():
    op.drop_table('questions')
    op.drop_table('subjects')
    op.drop_table('courses') 