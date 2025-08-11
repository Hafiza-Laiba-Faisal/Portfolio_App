import React, { useState, useEffect, useCallback } from 'react';
import { PlusCircle, Edit, Trash2, Github, ExternalLink, User, Code, Briefcase, LogOut } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const App = () => {
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('projects');
  const [projects, setProjects] = useState([]);
  const [skills, setSkills] = useState([]);
  const [experience, setExperience] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [modalType, setModalType] = useState('');
  const [editingItem, setEditingItem] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [deleteItem, setDeleteItem] = useState({ type: null, id: null });

  // Auth states
  const [authMode, setAuthMode] = useState('login');
  const [authForm, setAuthForm] = useState({ username: '', email: '', password: '' });

  // Form states
  const [projectForm, setProjectForm] = useState({
    title: '',
    description: '',
    tech_stack: '',
    github_url: '',
    live_url: '',
  });
  const [skillForm, setSkillForm] = useState({
    name: '',
    category: '',
    proficiency: 1,
  });
  const [experienceForm, setExperienceForm] = useState({
    company: '',
    position: '',
    description: '',
    start_date: '',
    end_date: '',
  });

  const apiCall = useCallback(async (endpoint, options = {}) => {
    const token = user?.token;
    const config = {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    };

    if (token && !endpoint.includes('auth')) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, config);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API call failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error:', error.message);
      if (error.message.includes('401')) {
        handleLogout();
        alert('Session expired. Please log in again.');
      }
      throw error;
    }
  }, [user?.token]);

  const loadData = useCallback(async () => {
    if (!user) return;
    
    try {
      const [projectsData, skillsData, experienceData] = await Promise.all([
        apiCall('/projects'),
        apiCall('/skills'),
        apiCall('/experience'),
      ]);
      console.log('Projects:', projectsData);
      console.log('Skills:', skillsData);
      console.log('Experience:', experienceData);
      setProjects(projectsData || []);
      setSkills(skillsData || []);
      setExperience(experienceData || []);
    } catch (error) {
      console.error('Failed to load data:', error.message);
      alert(`Failed to load data: ${error.message}`);
    }
  }, [apiCall, user]);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setUser({ token });
    }
  }, []);

  useEffect(() => {
    if (user) {
      loadData();
    }
  }, [user, loadData]);

  const handleAuth = async (e) => {
    e.preventDefault();
    try {
      if (authMode === 'register') {
        await apiCall('/auth/register', {
          method: 'POST',
          body: JSON.stringify(authForm),
        });
        alert('Registration successful! Please login.');
        setAuthMode('login');
      } else {
        const response = await apiCall('/auth/login', {
          method: 'POST',
          body: JSON.stringify({ username: authForm.username, password: authForm.password }),
        });
        localStorage.setItem('token', response.access_token);
        setUser({ token: response.access_token });
      }
    } catch (error) {
      alert('Authentication failed: ' + error.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
    setProjects([]);
    setSkills([]);
    setExperience([]);
  };

  const handleSubmit = async (e, type) => {
    e.preventDefault();
    try {
      const forms = { project: projectForm, skill: skillForm, experience: experienceForm };
      const endpoints = { project: '/projects', skill: '/skills', experience: '/experience' };
      const form = forms[type];
      const endpoint = endpoints[type];

      if (editingItem) {
        await apiCall(`${endpoint}/${editingItem.id}`, {
          method: 'PUT',
          body: JSON.stringify(form),
        });
      } else {
        await apiCall(endpoint, {
          method: 'POST',
          body: JSON.stringify(form),
        });
      }

      loadData();
      closeModal();
    } catch (error) {
      alert('Operation failed: ' + error.message);
    }
  };

  const handleDelete = (type, id) => {
    setDeleteItem({ type, id });
    setShowConfirmModal(true);
  };

  const confirmDelete = async () => {
    try {
      const { type, id } = deleteItem;
      const endpoints = { project: '/projects', skill: '/skills', experience: '/experience' };
      await apiCall(`${endpoints[type]}/${id}`, { method: 'DELETE' });
      loadData();
      setShowConfirmModal(false);
      setDeleteItem({ type: null, id: null });
    } catch (error) {
      alert('Delete failed: ' + error.message);
    }
  };

  const openModal = (type, item = null) => {
    setModalType(type);
    setEditingItem(item);
    setShowModal(true);

    if (item) {
      if (type === 'project') setProjectForm(item);
      if (type === 'skill') setSkillForm(item);
      if (type === 'experience') setExperienceForm(item);
    } else {
      setProjectForm({ title: '', description: '', tech_stack: '', github_url: '', live_url: '' });
      setSkillForm({ name: '', category: '', proficiency: 1 });
      setExperienceForm({ company: '', position: '', description: '', start_date: '', end_date: '' });
    }
  };

  const closeModal = () => {
    setShowModal(false);
    setModalType('');
    setEditingItem(null);
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-md p-6">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-gray-900">Portfolio Manager</h1>
            <p className="text-gray-600">Sign in to manage your portfolio</p>
          </div>

          <div className="flex mb-4">
            <button
              onClick={() => setAuthMode('login')}
              className={`flex-1 py-2 px-4 text-center rounded-l-lg ${
                authMode === 'login' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => setAuthMode('register')}
              className={`flex-1 py-2 px-4 text-center rounded-r-lg ${
                authMode === 'register' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
              }`}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleAuth}>
            <div className="space-y-4">
              <input
                type="text"
                placeholder="Username"
                value={authForm.username}
                onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                required
              />
              {authMode === 'register' && (
                <input
                  type="email"
                  placeholder="Email"
                  value={authForm.email}
                  onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                  required
                />
              )}
              <input
                type="password"
                placeholder="Password"
                value={authForm.password}
                onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full mt-4 bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors"
            >
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">Portfolio Manager</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Welcome back!</span>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-1 text-gray-600 hover:text-gray-900"
              >
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-1 mb-6">
          {[
            { id: 'projects', label: 'Projects', icon: Code },
            { id: 'skills', label: 'Skills', icon: User },
            { id: 'experience', label: 'Experience', icon: Briefcase },
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 capitalize">{activeTab}</h2>
          <button
            onClick={() => openModal(activeTab.slice(0, -1))}
            className="flex items-center space-x-2 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors"
          >
            <PlusCircle size={16} />
            <span>Add {activeTab.slice(0, -1)}</span>
          </button>
        </div>

        {activeTab === 'projects' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project) => (
              <div key={project.id} className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">{project.title}</h3>
                  <div className="flex space-x-1">
                    <button
                      onClick={() => openModal('project', project)}
                      className="text-gray-400 hover:text-blue-500"
                    >
                      <Edit size={16} />
                    </button>
                    <button
                      onClick={() => handleDelete('project', project.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
                <p className="text-gray-600 text-sm mb-3">{project.description}</p>
                <div className="mb-3">
                  <span className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">
                    {project.tech_stack}
                  </span>
                </div>
                <div className="flex space-x-3">
                  <a
                    href={project.github_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-500 hover:text-gray-700"
                  >
                    <Github size={16} />
                  </a>
                  {project.live_url && (
                    <a
                      href={project.live_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-500 hover:text-gray-700"
                    >
                      <ExternalLink size={16} />
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'skills' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {skills.map((skill) => (
              <div key={skill.id} className="bg-white rounded-lg shadow-sm border p-4">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900">{skill.name}</h3>
                    <p className="text-sm text-gray-600">{skill.category}</p>
                    <div className="mt-2">
                      <div className="flex items-center space-x-2">
                        <div className="bg-gray-200 rounded-full h-2 flex-1">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${(skill.proficiency / 5) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500">{skill.proficiency}/5</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-1 ml-4">
                    <button
                      onClick={() => openModal('skill', skill)}
                      className="text-gray-400 hover:text-blue-500"
                    >
                      <Edit size={16} />
                    </button>
                    <button
                      onClick={() => handleDelete('skill', skill.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'experience' && (
          <div className="space-y-6">
            {experience.map((exp) => (
              <div key={exp.id} className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900">{exp.position}</h3>
                    <p className="text-blue-600 font-medium">{exp.company}</p>
                    <p className="text-sm text-gray-500 mt-1">
                      {exp.start_date} - {exp.end_date || 'Present'}
                    </p>
                    <p className="text-gray-600 mt-3">{exp.description}</p>
                  </div>
                  <div className="flex space-x-1 ml-4">
                    <button
                      onClick={() => openModal('experience', exp)}
                      className="text-gray-400 hover:text-blue-500"
                    >
                      <Edit size={16} />
                    </button>
                    <button
                      onClick={() => handleDelete('experience', exp.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                {editingItem ? 'Edit' : 'Add'} {modalType}
              </h3>
              <button onClick={closeModal} className="text-gray-400 hover:text-gray-600">
                ✕
              </button>
            </div>

            {modalType === 'project' && (
              <form onSubmit={(e) => handleSubmit(e, 'project')}>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Project Title"
                    value={projectForm.title}
                    onChange={(e) => setProjectForm({ ...projectForm, title: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <textarea
                    placeholder="Description"
                    value={projectForm.description}
                    onChange={(e) => setProjectForm({ ...projectForm, description: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 h-20"
                    required
                  />
                  <input
                    type="text"
                    placeholder="Tech Stack (e.g., React, Node.js)"
                    value={projectForm.tech_stack}
                    onChange={(e) => setProjectForm({ ...projectForm, tech_stack: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <input
                    type="url"
                    placeholder="GitHub URL"
                    value={projectForm.github_url}
                    onChange={(e) => setProjectForm({ ...projectForm, github_url: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <input
                    type="url"
                    placeholder="Live URL (optional)"
                    value={projectForm.live_url}
                    onChange={(e) => setProjectForm({ ...projectForm, live_url: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    type="button"
                    onClick={closeModal}
                    className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                  >
                    {editingItem ? 'Update' : 'Create'}
                  </button>
                </div>
              </form>
            )}

            {modalType === 'skill' && (
              <form onSubmit={(e) => handleSubmit(e, 'skill')}>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Skill Name"
                    value={skillForm.name}
                    onChange={(e) => setSkillForm({ ...skillForm, name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <select
                    value={skillForm.category}
                    onChange={(e) => setSkillForm({ ...skillForm, category: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  >
                    <option value="">Select Category</option>
                    <option value="Frontend">Frontend</option>
                    <option value="Backend">Backend</option>
                    <option value="Database">Database</option>
                    <option value="DevOps">DevOps</option>
                    <option value="Mobile">Mobile</option>
                    <option value="Design">Design</option>
                    <option value="Other">Other</option>
                  </select>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Proficiency Level: {skillForm.proficiency}/5
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={skillForm.proficiency}
                      onChange={(e) => setSkillForm({ ...skillForm, proficiency: parseInt(e.target.value) })}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Beginner</span>
                      <span>Expert</span>
                    </div>
                  </div>
                </div>
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    type="button"
                    onClick={closeModal}
                    className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                  >
                    {editingItem ? 'Update' : 'Create'}
                  </button>
                </div>
              </form>
            )}

            {modalType === 'experience' && (
              <form onSubmit={(e) => handleSubmit(e, 'experience')}>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Company Name"
                    value={experienceForm.company}
                    onChange={(e) => setExperienceForm({ ...experienceForm, company: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <input
                    type="text"
                    placeholder="Position/Role"
                    value={experienceForm.position}
                    onChange={(e) => setExperienceForm({ ...experienceForm, position: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    required
                  />
                  <textarea
                    placeholder="Job Description"
                    value={experienceForm.description}
                    onChange={(e) => setExperienceForm({ ...experienceForm, description: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 h-20"
                    required
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <input
                      type="text"
                      placeholder="Start Date (e.g., Jan 2023)"
                      value={experienceForm.start_date}
                      onChange={(e) => setExperienceForm({ ...experienceForm, start_date: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                      required
                    />
                    <input
                      type="text"
                      placeholder="End Date (leave empty if current)"
                      value={experienceForm.end_date}
                      onChange={(e) => setExperienceForm({ ...experienceForm, end_date: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    type="button"
                    onClick={closeModal}
                    className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                  >
                    {editingItem ? 'Update' : 'Create'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}

      {/* Confirm Delete Modal */}
      {showConfirmModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-semibold mb-4">Confirm Deletion</h3>
            <p className="text-gray-600 mb-6">Are you sure you want to delete this item?</p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowConfirmModal(false)}
                className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;