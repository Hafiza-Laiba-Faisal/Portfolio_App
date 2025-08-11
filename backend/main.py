from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import bcrypt
from contextlib import contextmanager
import os
# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./portfolio.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Portfolio API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Portfolio API. Visit /docs for API documentation."}

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    tech_stack = Column(String)
    github_url = Column(String)
    live_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String)
    proficiency = Column(Integer)

class Experience(Base):
    __tablename__ = "experience"
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String)
    position = Column(String)
    description = Column(Text)
    start_date = Column(String)
    end_date = Column(String, nullable=True)

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ProjectCreate(BaseModel):
    title: str
    description: str
    tech_stack: str
    github_url: str
    live_url: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    title: str
    description: str
    tech_stack: str
    github_url: str
    live_url: Optional[str]
    created_at: datetime

class SkillCreate(BaseModel):
    name: str
    category: str
    proficiency: int

class SkillResponse(BaseModel):
    id: int
    name: str
    category: str
    proficiency: int

class ExperienceCreate(BaseModel):
    company: str
    position: str
    description: str
    start_date: str
    end_date: Optional[str] = None

class ExperienceResponse(BaseModel):
    id: int
    company: str
    position: str
    description: str
    start_date: str
    end_date: Optional[str]

class Token(BaseModel):
    access_token: str
    token_type: str

# Create tables
Base.metadata.create_all(bind=engine)

# Dependencies
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

security = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Authentication Routes
@app.post("/auth/register", response_model=dict)
def register(user: UserCreate, db: Session = Depends(get_db)):
    with get_db() as db_session:
        db_user = db_session.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        db_user = db_session.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = hash_password(user.password)
        db_user = User(
            username=user.username,
            email=user.email,
            password_hash=hashed_password
        )
        db_session.add(db_user)
        db_session.commit()
        
        return {"message": "User created successfully"}

@app.post("/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    with get_db() as db_session:
        db_user = db_session.query(User).filter(User.username == user.username).first()
        if not db_user or not verify_password(user.password, db_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

# Project Routes
@app.get("/projects", response_model=List[ProjectResponse])
def get_projects(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    with get_db() as db_session:
        return db_session.query(Project).all()

@app.post("/projects", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_project = Project(**project.dict())
        db_session.add(db_project)
        db_session.commit()
        db_session.refresh(db_project)
        return db_project

@app.put("/projects/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int,
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_project = db_session.query(Project).filter(Project.id == project_id).first()
        if not db_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        for key, value in project.dict().items():
            setattr(db_project, key, value)
        
        db_session.commit()
        db_session.refresh(db_project)
        return db_project

@app.delete("/projects/{project_id}")
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_project = db_session.query(Project).filter(Project.id == project_id).first()
        if not db_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        db_session.delete(db_project)
        db_session.commit()
        return {"message": "Project deleted successfully"}

# Skills Routes
@app.get("/skills", response_model=List[SkillResponse])
def get_skills(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    with get_db() as db_session:
        return db_session.query(Skill).all()

@app.post("/skills", response_model=SkillResponse)
def create_skill(
    skill: SkillCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_skill = Skill(**skill.dict())
        db_session.add(db_skill)
        db_session.commit()
        db_session.refresh(db_skill)
        return db_skill

# Experience Routes
@app.get("/experience", response_model=List[ExperienceResponse])
def get_experience(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    with get_db() as db_session:
        return db_session.query(Experience).all()

@app.post("/experience", response_model=ExperienceResponse)
def create_experience(
    experience: ExperienceCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_experience = Experience(**experience.dict())
        db_session.add(db_experience)
        db_session.commit()
        db_session.refresh(db_experience)
        return db_experience
    
@app.put("/skills/{skill_id}", response_model=SkillResponse)
def update_skill(
    skill_id: int,
    skill: SkillCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_skill = db_session.query(Skill).filter(Skill.id == skill_id).first()
        if not db_skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        for key, value in skill.dict().items():
            setattr(db_skill, key, value)
        
        db_session.commit()
        db_session.refresh(db_skill)
        return db_skill

@app.delete("/skills/{skill_id}")
def delete_skill(
    skill_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_skill = db_session.query(Skill).filter(Skill.id == skill_id).first()
        if not db_skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        db_session.delete(db_skill)
        db_session.commit()
        return {"message": "Skill deleted successfully"}

@app.put("/experience/{experience_id}", response_model=ExperienceResponse)
def update_experience(
    experience_id: int,
    experience: ExperienceCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_experience = db_session.query(Experience).filter(Experience.id == experience_id).first()
        if not db_experience:
            raise HTTPException(status_code=404, detail="Experience not found")
        
        for key, value in experience.dict().items():
            setattr(db_experience, key, value)
        
        db_session.commit()
        db_session.refresh(db_experience)
        return db_experience

@app.delete("/experience/{experience_id}")
def delete_experience(
    experience_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    with get_db() as db_session:
        db_experience = db_session.query(Experience).filter(Experience.id == experience_id).first()
        if not db_experience:
            raise HTTPException(status_code=404, detail="Experience not found")
        
        db_session.delete(db_experience)
        db_session.commit()
        return {"message": "Experience deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)