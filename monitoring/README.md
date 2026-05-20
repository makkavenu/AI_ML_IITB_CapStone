# Monitoring run modes

This project includes Prometheus and Grafana in the default Docker Compose stack.

## Local Windows / Docker Desktop

Run only the cross-platform services:

```bash
docker compose up --build
```

This starts `api`, `ui`, `prometheus`, and `grafana`. The default Prometheus
config scrapes only `api:8000` and `prometheus:9090`. The Linux host exporters
`node-exporter` and `cadvisor` are behind the `linux-host-monitoring` profile and
are intentionally disabled by default because Docker Desktop on Windows/WSL2 does
not support the Linux bind mount propagation used by node-exporter (`/:/host:ro,rslave`).

## EC2 Linux demo

Set this in `.env` on EC2 so Prometheus also scrapes the Linux host exporters:

```env
PROMETHEUS_CONFIG_FILE=./monitoring/prometheus/prometheus.ec2.yml
```

Run the full stack, including host/container monitoring:

```bash
docker compose --profile linux-host-monitoring up -d --build
```

With CloudWatch logging enabled on EC2:

```bash
docker compose -f docker-compose.yml -f docker-compose.ec2-logs.yml --profile linux-host-monitoring up -d --build
```

With `PROMETHEUS_CONFIG_FILE=./monitoring/prometheus/prometheus.ec2.yml`, Prometheus
will scrape `api:8000`, `prometheus:9090`, `node-exporter:9100`, and `cadvisor:8080`.
