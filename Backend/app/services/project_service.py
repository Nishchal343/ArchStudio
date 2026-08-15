from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Project


async def get_owned_project(db: AsyncSession, project_id, user_id) -> Project | None:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == user_id))
    return result.scalar_one_or_none()
