from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..dependencies import get_current_user
from ..models import UserProfile
from ..schemas import ProfileResponse

router = APIRouter(tags=["users"])


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    user = current_user["raw"]
    return {"id": str(user.id), "email": user.email, "email_confirmed": bool(user.email_confirmed_at), "metadata": user.user_metadata or {}}


@router.get("/me/profile", response_model=ProfileResponse)
async def profile(current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserProfile).where(UserProfile.supabase_user_id == current_user["id"]))
    profile = result.scalar_one_or_none()
    if not profile:
        profile = UserProfile(supabase_user_id=current_user["id"], display_name=(current_user["raw"].user_metadata or {}).get("name"))
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    return profile

