# AI Trading System - Deployment Strategies

## Overview

This document provides comprehensive deployment strategies for the AI Trading System, covering different environments, deployment methods, and best practices for production deployment.

## Deployment Environments

### Development Environment

#### Local Development
```bash
# Clone repository
git clone <repository-url>
cd TradingBOT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.template .env
# Edit .env with your configuration

# Initialize database
python -c "from src.config.database import get_database_manager; get_database_manager().initialize_database()"

# Start development server
python src/main.py
```

#### Docker Development
```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install development dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Create data directory
RUN mkdir -p data logs

# Set permissions
RUN chmod +x src/main.py

# Expose port
EXPOSE 8050

# Run in development mode
CMD ["python", "src/main.py", "--dev"]
```

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  tradingbot-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8050:8050"
    volumes:
      - .:/app
      - /app/venv
    environment:
      - MODE=DEMO
      - LOG_LEVEL=DEBUG
    depends_on:
      - redis-dev
      - postgres-dev

  redis-dev:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data

  postgres-dev:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: tradingbot_dev
      POSTGRES_USER: tradingbot
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data

volumes:
  redis_dev_data:
  postgres_dev_data:
```

### Staging Environment

#### Staging Server Setup
```bash
# Server preparation
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install redis-server postgresql-client nginx

# Create application user
sudo useradd -m -s /bin/bash tradingbot
sudo su - tradingbot

# Clone and setup application
git clone <repository-url>
cd TradingBOT
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup environment
cp .env.staging .env
# Configure staging environment variables

# Initialize database
python -c "from src.config.database import get_database_manager; get_database_manager().initialize_database()"
```

#### Staging Configuration
```yaml
# config/staging.yaml
database:
  url: "postgresql://tradingbot:staging_password@localhost:5432/tradingbot_staging"
  pool_size: 10
  max_overflow: 20

redis:
  url: "redis://localhost:6379/1"
  max_connections: 20

trading:
  mode: "DEMO"
  max_positions: 5
  position_size_percent: 5.0
  risk_limit_percent: 1.0

monitoring:
  log_level: "INFO"
  enable_alerts: true
  alert_email: "staging-alerts@company.com"

security:
  api_rate_limit: 100
  session_timeout: 3600
```

#### Staging Deployment Script
```bash
#!/bin/bash
# deploy-staging.sh

set -e

echo "Starting staging deployment..."

# Pull latest code
git pull origin develop

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Run database migrations
python scripts/migrate.py

# Run tests
pytest tests/ --cov=src

# Restart services
sudo systemctl restart tradingbot-staging
sudo systemctl restart nginx

# Health check
sleep 10
curl -f http://localhost:8050/health || exit 1

echo "Staging deployment completed successfully!"
```

### Production Environment

#### Production Server Setup
```bash
# Server preparation
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install redis-server postgresql-client nginx certbot

# Create application user
sudo useradd -m -s /bin/bash tradingbot
sudo su - tradingbot

# Clone and setup application
git clone <repository-url>
cd TradingBOT
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup environment
cp .env.production .env
# Configure production environment variables

# Initialize database
python -c "from src.config.database import get_database_manager; get_database_manager().initialize_database()"
```

#### Production Configuration
```yaml
# config/production.yaml
database:
  url: "postgresql://tradingbot:production_password@localhost:5432/tradingbot_prod"
  pool_size: 20
  max_overflow: 30

redis:
  url: "redis://localhost:6379/0"
  max_connections: 50

trading:
  mode: "LIVE"
  max_positions: 10
  position_size_percent: 10.0
  risk_limit_percent: 2.0

monitoring:
  log_level: "WARNING"
  enable_alerts: true
  alert_email: "production-alerts@company.com"
  alert_sms: "+1234567890"

security:
  api_rate_limit: 1000
  session_timeout: 1800
  ssl_required: true
```

## Deployment Methods

### 1. Blue-Green Deployment

#### Blue-Green Setup
```bash
#!/bin/bash
# blue-green-deploy.sh

set -e

CURRENT_COLOR=$(cat /opt/tradingbot/current_color)
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "Current color: $CURRENT_COLOR"
echo "Deploying to: $NEW_COLOR"

# Deploy to new environment
sudo systemctl stop tradingbot-$NEW_COLOR || true
sudo systemctl start tradingbot-$NEW_COLOR

# Wait for health check
sleep 30
curl -f http://localhost:805$([ "$NEW_COLOR" = "blue" ] && echo "1" || echo "2")/health || exit 1

# Switch traffic
sudo nginx -s reload

# Update current color
echo $NEW_COLOR > /opt/tradingbot/current_color

# Stop old environment
sudo systemctl stop tradingbot-$CURRENT_COLOR

echo "Blue-green deployment completed successfully!"
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/tradingbot
upstream tradingbot_blue {
    server 127.0.0.1:8051;
}

upstream tradingbot_green {
    server 127.0.0.1:8052;
}

map $cookie_color $backend {
    default $upstream;
    blue tradingbot_blue;
    green tradingbot_green;
}

server {
    listen 80;
    server_name tradingbot.com;

    location / {
        proxy_pass http://$backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Canary Deployment

#### Canary Setup
```bash
#!/bin/bash
# canary-deploy.sh

set -e

CANARY_PERCENTAGE=${1:-10}

echo "Starting canary deployment with $CANARY_PERCENTAGE% traffic"

# Deploy canary version
sudo systemctl stop tradingbot-canary || true
sudo systemctl start tradingbot-canary

# Wait for health check
sleep 30
curl -f http://localhost:8053/health || exit 1

# Update nginx configuration for canary traffic
sudo nginx -s reload

# Monitor canary for 10 minutes
echo "Monitoring canary deployment for 10 minutes..."
sleep 600

# Check canary metrics
CANARY_ERROR_RATE=$(curl -s http://localhost:8053/metrics | grep error_rate | cut -d' ' -f2)
CANARY_RESPONSE_TIME=$(curl -s http://localhost:8053/metrics | grep response_time | cut -d' ' -f2)

if (( $(echo "$CANARY_ERROR_RATE < 0.01" | bc -l) )) && (( $(echo "$CANARY_RESPONSE_TIME < 2000" | bc -l) )); then
    echo "Canary deployment successful. Promoting to production..."
    
    # Promote canary to production
    sudo systemctl stop tradingbot-production
    sudo systemctl start tradingbot-production
    
    # Update nginx to route all traffic to production
    sudo nginx -s reload
    
    # Stop canary
    sudo systemctl stop tradingbot-canary
    
    echo "Canary deployment promoted to production successfully!"
else
    echo "Canary deployment failed. Rolling back..."
    sudo systemctl stop tradingbot-canary
    sudo nginx -s reload
    exit 1
fi
```

#### Canary Nginx Configuration
```nginx
# /etc/nginx/sites-available/tradingbot-canary
upstream tradingbot_production {
    server 127.0.0.1:8050;
}

upstream tradingbot_canary {
    server 127.0.0.1:8053;
}

split_clients $remote_addr $backend {
    10% tradingbot_canary;
    * tradingbot_production;
}

server {
    listen 80;
    server_name tradingbot.com;

    location / {
        proxy_pass http://$backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Rolling Deployment

#### Rolling Update Script
```bash
#!/bin/bash
# rolling-deploy.sh

set -e

TOTAL_INSTANCES=3
INSTANCE_PREFIX="tradingbot-instance"

echo "Starting rolling deployment..."

for i in $(seq 1 $TOTAL_INSTANCES); do
    INSTANCE_NAME="$INSTANCE_PREFIX-$i"
    PORT=$((8050 + i))
    
    echo "Updating instance $INSTANCE_NAME on port $PORT"
    
    # Stop instance
    sudo systemctl stop $INSTANCE_NAME || true
    
    # Deploy new version
    sudo systemctl start $INSTANCE_NAME
    
    # Wait for health check
    sleep 30
    curl -f http://localhost:$PORT/health || exit 1
    
    echo "Instance $INSTANCE_NAME updated successfully"
    
    # Wait between updates
    sleep 60
done

echo "Rolling deployment completed successfully!"
```

#### Load Balancer Configuration
```nginx
# /etc/nginx/sites-available/tradingbot-load-balancer
upstream tradingbot_backend {
    server 127.0.0.1:8051;
    server 127.0.0.1:8052;
    server 127.0.0.1:8053;
}

server {
    listen 80;
    server_name tradingbot.com;

    location / {
        proxy_pass http://tradingbot_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Container Deployment

### Docker Production

#### Production Dockerfile
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Create non-root user
RUN useradd -m -u 1000 tradingbot
RUN chown -R tradingbot:tradingbot /app
USER tradingbot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Expose port
EXPOSE 8050

# Run application
CMD ["python", "src/main.py"]
```

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  tradingbot:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8050:8050"
    environment:
      - MODE=LIVE
      - DATABASE_URL=postgresql://tradingbot:${DB_PASSWORD}@postgres:5432/tradingbot
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: tradingbot
      POSTGRES_USER: tradingbot
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.1'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - tradingbot
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Kubernetes Manifests
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tradingbot
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tradingbot-config
  namespace: tradingbot
data:
  config.yaml: |
    database:
      url: "postgresql://tradingbot:${DB_PASSWORD}@postgres:5432/tradingbot"
    redis:
      url: "redis://redis:6379"
    trading:
      mode: "LIVE"
      max_positions: 10
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tradingbot-secrets
  namespace: tradingbot
type: Opaque
data:
  db-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tradingbot
  namespace: tradingbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tradingbot
  template:
    metadata:
      labels:
        app: tradingbot
    spec:
      containers:
      - name: tradingbot
        image: tradingbot:latest
        ports:
        - containerPort: 8050
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tradingbot-secrets
              key: db-password
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: tradingbot-secrets
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8050
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8050
          initialDelaySeconds: 5
          periodSeconds: 5
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tradingbot-service
  namespace: tradingbot
spec:
  selector:
    app: tradingbot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8050
  type: LoadBalancer
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tradingbot-ingress
  namespace: tradingbot
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - tradingbot.com
    secretName: tradingbot-tls
  rules:
  - host: tradingbot.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tradingbot-service
            port:
              number: 80
```

#### Kubernetes Deployment Script
```bash
#!/bin/bash
# k8s-deploy.sh

set -e

NAMESPACE="tradingbot"
IMAGE_TAG=${1:-latest}

echo "Deploying to Kubernetes with image tag: $IMAGE_TAG"

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Update deployment with new image
kubectl set image deployment/tradingbot tradingbot=tradingbot:$IMAGE_TAG -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/tradingbot -n $NAMESPACE

# Apply service and ingress
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

echo "Kubernetes deployment completed successfully!"
```

## Cloud Deployment

### AWS Deployment

#### AWS ECS
```json
{
  "family": "tradingbot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "tradingbot",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/tradingbot:latest",
      "portMappings": [
        {
          "containerPort": 8050,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODE",
          "value": "LIVE"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:tradingbot/db-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tradingbot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8050/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### AWS Deployment Script
```bash
#!/bin/bash
# aws-deploy.sh

set -e

AWS_REGION="us-east-1"
ECR_REPOSITORY="tradingbot"
ECS_CLUSTER="tradingbot-cluster"
ECS_SERVICE="tradingbot-service"

echo "Starting AWS deployment..."

# Build and push Docker image
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker build -t $ECR_REPOSITORY .
docker tag $ECR_REPOSITORY:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPOSITORY:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPOSITORY:latest

# Update ECS service
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --force-new-deployment --region $AWS_REGION

# Wait for deployment to complete
aws ecs wait services-stable --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION

echo "AWS deployment completed successfully!"
```

### Google Cloud Platform

#### GCP Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: tradingbot
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/tradingbot:latest
        ports:
        - containerPort: 8050
        env:
        - name: MODE
          value: "LIVE"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tradingbot-secrets
              key: database-url
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8050
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### GCP Deployment Script
```bash
#!/bin/bash
# gcp-deploy.sh

set -e

PROJECT_ID="your-project-id"
SERVICE_NAME="tradingbot"
REGION="us-central1"

echo "Starting GCP deployment..."

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10

echo "GCP deployment completed successfully!"
```

## Monitoring and Observability

### Production Monitoring

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tradingbot'
    static_configs:
      - targets: ['localhost:8050']
    metrics_path: /metrics
    scrape_interval: 5s
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "TradingBot Dashboard",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"tradingbot\"}",
            "legendFormat": "System Status"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds{job=\"tradingbot\"}",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Log Management

#### ELK Stack Configuration
```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## Backup and Recovery

### Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup-database.sh

set -e

BACKUP_DIR="/backups/tradingbot"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="tradingbot"
DB_USER="tradingbot"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME > $BACKUP_DIR/tradingbot_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/tradingbot_$DATE.sql

# Upload to S3 (if using AWS)
aws s3 cp $BACKUP_DIR/tradingbot_$DATE.sql.gz s3://tradingbot-backups/database/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Database backup completed: tradingbot_$DATE.sql.gz"
```

#### Recovery Script
```bash
#!/bin/bash
# restore-database.sh

set -e

BACKUP_FILE=$1
DB_NAME="tradingbot"
DB_USER="tradingbot"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Restoring database from $BACKUP_FILE..."

# Stop application
sudo systemctl stop tradingbot

# Restore database
gunzip -c $BACKUP_FILE | psql -h localhost -U $DB_USER -d $DB_NAME

# Start application
sudo systemctl start tradingbot

echo "Database restore completed successfully!"
```

## Security Considerations

### SSL/TLS Configuration

#### Let's Encrypt Setup
```bash
#!/bin/bash
# setup-ssl.sh

set -e

DOMAIN="tradingbot.com"
EMAIL="admin@tradingbot.com"

# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d $DOMAIN --email $EMAIL --agree-tos --non-interactive

# Setup auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### Nginx SSL Configuration
```nginx
# /etc/nginx/sites-available/tradingbot-ssl
server {
    listen 443 ssl http2;
    server_name tradingbot.com;
    
    ssl_certificate /etc/letsencrypt/live/tradingbot.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tradingbot.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    location / {
        proxy_pass http://localhost:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name tradingbot.com;
    return 301 https://$server_name$request_uri;
}
```

## Conclusion

This deployment strategies guide provides comprehensive information for deploying the AI Trading System across different environments and platforms. Choose the deployment method that best fits your requirements and infrastructure.

Key considerations:
- **Environment**: Development, staging, or production
- **Deployment Method**: Blue-green, canary, or rolling
- **Platform**: On-premises, cloud, or hybrid
- **Monitoring**: Comprehensive observability
- **Security**: SSL/TLS and access controls
- **Backup**: Automated backup and recovery

For additional support or questions, please refer to the documentation or contact the development team.
