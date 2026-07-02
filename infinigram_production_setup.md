# Infinigram API Production Setup Guide

This guide will help you set up a production deployment of the Infinigram API for your framing audit system.

## Prerequisites

1. **Docker and Docker Compose** installed
2. **Sufficient disk space** (>100GB for indices)
3. **AWS CLI** (optional, for downloading additional indices)

## Step 1: Download Required Indices

Navigate to your infinigram-api directory:
```bash
cd /Users/a1234567/olmo/infinigram-api
```

Download the base index (this will take some time and bandwidth):
```bash
./bin/download-local-infini-gram-array.sh
```

This script will:
- Download `v4_pileval_llama` (~100GB)
- Create symlinks for all OLMo indices to use this base index
- Set up the directory structure needed by docker-compose

## Step 2: Start the Production Services

Start all services using Docker Compose:
```bash
docker-compose up -d
```

This will start:
- **API service** on port 8080
- **Attribution worker** on port 8081  
- **PostgreSQL** queue on port 5432
- **Redis** cache on port 6379
- **Nginx proxy** for load balancing

Check that services are running:
```bash
docker-compose ps
```

## Step 3: Verify API is Working

Test the API endpoint:
```bash
curl "http://localhost:8080/v4_pileval_llama/count?phrase=hello%20world"
```

You should get a JSON response with a count value.

## Step 4: Configure for OLMo Index

Your test script uses `v4_olmo-2-1124-13b-instruct_llama` but the available indices are:
- `v4-olmo-2-1124-13b-anneal-adapt`
- `v4-olmo-2-0325-32b-anneal-adapt`

Update your test script to use one of these available indices.

## Step 5: Run Your Framing Audit Test

Once the API is running, you can test it with your script:
```bash
cd /Users/a1234567/olmo/framing_audit
python test_olmotrace_style.py
```

## Troubleshooting

### Services Won't Start
- Check Docker is running: `docker info`
- Check ports aren't in use: `lsof -i :8080`
- View logs: `docker-compose logs api`

### Index Download Issues
- Ensure good internet connection (100GB+ download)
- Check AWS credentials if using private indices
- Verify disk space: `df -h`

### API Connection Issues
- Verify API is responding: `curl http://localhost:8080/health`
- Check firewall settings
- Review nginx proxy logs: `docker-compose logs proxy`

## Production Considerations

1. **Resource Requirements**:
   - Memory: 16GB+ recommended
   - CPU: 8+ cores for good performance
   - Storage: 200GB+ for multiple indices

2. **Scaling**:
   - Increase attribution worker replicas in docker-compose.yaml
   - Add load balancer if running multiple API instances
   - Consider SSD storage for better index performance

3. **Monitoring**:
   - Check service health endpoints
   - Monitor Docker container resource usage
   - Set up log aggregation for production

4. **Security**:
   - Change default passwords in docker-compose.yaml
   - Add authentication if exposing externally
   - Use HTTPS in production

## Index Management

To add additional OLMo indices:
1. Download index files to `./infinigram-array/`
2. Add volume mounts to `docker-compose.yaml`
3. Update index mappings in `api/src/infinigram/index_mappings.py`
4. Restart services: `docker-compose restart`