# Chroma Vector Database Deployment

This directory contains files for deploying a Chroma vector database using Docker Compose, which can be used as the vector store for the LangChain RAG system.

## Prerequisites

- Docker and Docker Compose installed on your system
- Python 3.8+ (for running the credential generation script)

## Setup Instructions

1. Generate authentication credentials:

```bash
# Make the script executable
chmod +x generate_credentials.py

# Run the script to generate credentials
./generate_credentials.py
```

This will create a `chroma_auth.json` file with secure random tokens. **Save these tokens** - you'll need them to connect to your database.

Alternatively, you can copy the example file and modify it:

```bash
cp chroma_auth.json.example chroma_auth.json
# Edit the file to replace placeholder tokens with secure ones
```

2. Start the Chroma database:

```bash
docker-compose up -d
```

3. Verify it's running:

```bash
docker-compose ps
docker-compose logs
```

## Connecting to Chroma

Use the example script in `examples/chroma_docker_connection.py` to connect to your Chroma database:

1. Update the `CHROMA_TOKEN` value with your admin token from `chroma_auth.json`
2. Set your OpenAI API key
3. Run the script:

```bash
python examples/chroma_docker_connection.py
```

## Configuration Options

You can modify the `docker-compose.yml` file to customize your deployment:

- Change the port mapping if port 8000 is already in use
- Adjust environment variables
- Configure persistence options

## Stopping the Database

To stop the database:

```bash
docker-compose down
```

To completely remove the data volumes:

```bash
docker-compose down -v
```

## Troubleshooting

- **Connection issues**: Make sure Docker is running and the container is healthy
- **Authentication errors**: Check that you're using the correct token
- **Permission problems**: Ensure your user has proper permissions for Docker operations

## Security Notes

- Keep your `chroma_auth.json` file secure
- Use different tokens for different users/roles
- For production deployments, consider enabling SSL/TLS 