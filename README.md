# 🚀 Personal Portfolio API

A full-stack portfolio management system built with **FastAPI** and **React.js**. This project demonstrates modern web development practices including RESTful API design, JWT authentication, responsive frontend, and cloud deployment.


## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔐 Authentication
- User registration and login
- JWT token-based authentication
- Secure password hashing
- Protected routes and endpoints

### 📁 Portfolio Management
- **Projects**: CRUD operations for project showcase
- **Skills**: Manage technical skills with proficiency levels
- **Experience**: Track work experience and achievements
- **Real-time updates** across the application

### 🎨 Modern UI/UX
- Responsive design for all devices
- Clean, professional interface
- Interactive forms and modals
- Loading states and error handling

### 🌐 Production Ready
- RESTful API architecture
- Comprehensive error handling
- CORS configuration
- Database migrations
- Environment-based configuration

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (Production: PostgreSQL)
- **ORM**: SQLAlchemy
- **Authentication**: JWT (PyJWT)
- **Server**: Uvicorn

### Frontend
- **Framework**: React.js
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **State Management**: React Hooks
- **HTTP Client**: Fetch API

### Deployment
- **Backend**: Railway / Render
- **Frontend**: Vercel / Netlify
- **Database**: SQLite (local) / PostgreSQL (production)
- **Domain**: Custom domain support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1-Minute Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-api.git
cd portfolio-api

# Start backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# Start frontend (new terminal)
cd frontend
npm install
npm start
```

🎉 **Access your application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📦 Installation

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### Step-by-Step Setup

#### 1. Install Required Software
```bash
# Verify installations
python --version  # Should be 3.8+
node --version    # Should be 16+
npm --version     # Should be 8+
git --version     # Any recent version
```

#### 2. Clone and Setup Backend
```bash
git clone https://github.com/yourusername/portfolio-api.git
cd portfolio-api/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "SECRET_KEY=your-super-secret-key-change-in-production" > .env

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Setup Frontend
```bash
# Open new terminal
cd portfolio-api/frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📖 API Documentation

### Authentication Endpoints
```http
POST /auth/register
POST /auth/login
```

### Portfolio Endpoints
```http
GET    /projects        # Get all projects
POST   /projects        # Create project (Auth required)
PUT    /projects/{id}    # Update project (Auth required)
DELETE /projects/{id}    # Delete project (Auth required)

GET    /skills          # Get all skills
POST   /skills          # Create skill (Auth required)

GET    /experience      # Get all experience
POST   /experience      # Create experience (Auth required)
```

### Request/Response Examples

#### Register User
```json
POST /auth/register
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

#### Create Project
```json
POST /projects
Authorization: Bearer <your-jwt-token>
{
  "title": "E-commerce Website",
  "description": "Full-stack e-commerce platform with payment integration",
  "tech_stack": "React, Node.js, MongoDB, Stripe",
  "github_url": "https://github.com/johndoe/ecommerce",
  "live_url": "https://mystore.com"
}
```

### Interactive API Documentation
Visit http://localhost:8000/docs for Swagger UI with all endpoints, request/response schemas, and testing interface.

## 🌐 Deployment

### Backend Deployment (Railway)
1. Push code to GitHub
2. Connect GitHub repo to [Railway](https://railway.app)
3. Set environment variables:
   ```
   SECRET_KEY=your-production-secret-key
   ```
4. Deploy automatically

### Frontend Deployment (Vercel)
1. Update API base URL in React app
2. Push to GitHub
3. Connect repo to [Vercel](https://vercel.com)
4. Deploy automatically

### Environment Variables
```bash
# Backend (.env)
SECRET_KEY=your-super-secret-key
DATABASE_URL=sqlite:///./portfolio.db

# Frontend (.env)
REACT_APP_API_URL=https://your-api-domain.railway.app
```

## 📁 Project Structure

```
portfolio-api/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment variables
│   └── portfolio.db         # SQLite database (auto-created)
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── index.js        # Entry point
│   ├── package.json        # Node.js dependencies
│   └── .env                # Frontend environment variables
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🔧 Development

### Adding New Features
1. **Backend**: Add new routes in `main.py`
2. **Frontend**: Update React components
3. **Database**: Modify SQLAlchemy models

### Code Style
- **Backend**: Follow PEP 8 (Python style guide)
- **Frontend**: Use Prettier for formatting
- **Commits**: Use conventional commit messages

### Testing
```bash
# Backend testing
cd backend
pytest

# Frontend testing
cd frontend
npm test
```

## 🚨 Common Issues & Solutions

### Port Already in Use
```bash
# Kill process on port 8000
# Windows: netstat -ano | findstr :8000
# Mac/Linux: lsof -ti:8000 | xargs kill -9
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
```

### CORS Errors
Update CORS origins in `main.py`:
```python
allow_origins=["http://localhost:3000", "https://your-domain.com"]
```

## 📈 Performance & Scaling

### Current Limitations
- SQLite for development (single-user)
- No caching implemented
- Basic error logging

### Production Recommendations
- Upgrade to PostgreSQL
- Implement Redis for caching
- Add rate limiting
- Set up monitoring (e.g., Sentry)
- Use CDN for static assets

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup for Contributors
```bash
git clone https://github.com/yourusername/portfolio-api.git
cd portfolio-api
git checkout -b feature/your-feature

# Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

## 🎯 Roadmap

### Version 2.0 Features
- [ ] File upload for project images
- [ ] Contact form with email notifications
- [ ] Public portfolio view (no authentication)
- [ ] Blog/Articles section
- [ ] Testimonials management
- [ ] Analytics dashboard
- [ ] Export portfolio to PDF
- [ ] Multi-language support

### Technical Improvements
- [ ] Unit and integration tests
- [ ] CI/CD pipeline
- [ ] Docker containerization
- [ ] Rate limiting
- [ ] API versioning
- [ ] Comprehensive logging
- [ ] Performance monitoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used
- [React](https://reactjs.org/) - Frontend library
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Lucide](https://lucide.dev/) - Icon library

## 📞 Support

If you have any questions or run into issues:

1. **Check the documentation** above
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Join our community** discussions

---

**⭐ Star this repository if it helped you build your portfolio!**



Built with  by Hafiza Laiba Faisal


