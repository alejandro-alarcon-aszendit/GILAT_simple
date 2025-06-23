# Docker Deployment Guide

This guide explains how to deploy the GILAT Document Service using Docker Compose.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- OpenAI API key

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

## Services

### API Service (Backend)
- **Port**: 8000
- **Container**: `gilat_api`
- **Health Check**: Available at `/health`

### Frontend Service (UI)
- **Port**: 8501
- **Container**: `gilat_frontend`
- **Dependencies**: Connects to the API service

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional
- `API_AUTH_KEY`: API authentication key (leave empty to disable authentication)
- `JWT_SECRET_KEY`: Secret key for JWT token signing (required if using authentication)

## Data Persistence

- `./vector_db`: Vector embeddings and document chunks
- `./app.db`: SQLite database file

## Common Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

## Troubleshooting

### API Connection Issues
- Check API logs: `docker-compose logs api`
- Verify `OPENAI_API_KEY` is set in `.env`

### Authentication Issues
- If authentication is enabled (`API_AUTH_KEY` set), users must login in the Streamlit sidebar
- Click "Login" after entering API key, or "No Auth" if authentication is disabled
- Tokens expire after 24 hours - users will need to login again
- To disable authentication, remove or leave empty the `API_AUTH_KEY` in your `.env` file
- **JWT Secret**: Ensure `JWT_SECRET_KEY` is set to a secure random string in production

### Port Conflicts
If ports 8000 or 8501 are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Use port 9000 instead of 8000
```

## Security Features

### JWT Authentication
- **Secure Token-Based Auth**: Users login with API key to receive JWT tokens
- **No Direct API Key Exposure**: Frontend never stores or compares API keys directly
- **Token Expiration**: JWT tokens expire after 24 hours for security
- **Login Endpoint**: `/auth/login` validates API keys and returns secure tokens
- **Protected Endpoints**: All document operations require valid JWT tokens

### CORS Protection
- **Cross-Origin Requests**: Configured to allow requests from common frontend ports
- **Credentials Support**: Enables authentication headers in cross-origin requests
- **Method Restrictions**: Limited to essential HTTP methods (GET, POST, PUT, DELETE) 