# Standalone Mininet Docker Container Makefile
# Provides convenient commands for managing the container

CONTAINER_NAME = mininet-dev
IMAGE_NAME = mininet-standalone

.PHONY: help build run shell stop status logs clean test

# Default target
help:
	@echo "Standalone Mininet Docker Container"
	@echo ""
	@echo "Available commands:"
	@echo "  make build     - Build Docker image"
	@echo "  make run       - Run container interactively"
	@echo "  make shell     - Enter running container"
	@echo "  make stop      - Stop container"
	@echo "  make status    - Show container status"
	@echo "  make logs      - Show container logs"
	@echo "  make clean     - Remove container and image"
	@echo "  make test      - Run basic functionality tests"
	@echo "  make help      - Show this help"

# Build the Docker image
build:
	@echo "Building Docker image $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) .
	@echo "Build complete!"

# Run the container
run:
	@echo "Starting container $(CONTAINER_NAME)..."
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Stopping existing container..."; \
		docker stop $(CONTAINER_NAME); \
	fi
	@if docker ps -aq -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Removing existing container..."; \
		docker rm $(CONTAINER_NAME); \
	fi
	docker run -it --privileged \
		--name $(CONTAINER_NAME) \
		--network host \
		-v "$$(pwd)/projects:/app/projects" \
		-v "/lib/modules:/lib/modules:ro" \
		$(IMAGE_NAME)

# Enter running container
shell:
	@if ! docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Container not running. Starting it first..."; \
		make run; \
	else \
		echo "Entering container shell..."; \
		docker exec -it $(CONTAINER_NAME) /bin/bash; \
	fi

# Stop the container
stop:
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Stopping container..."; \
		docker stop $(CONTAINER_NAME); \
		echo "Container stopped"; \
	else \
		echo "Container is not running"; \
	fi

# Show container status
status:
	@echo "Container status:"
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "✓ Running"; \
		docker ps -f name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "✗ Not running"; \
	fi

# Show container logs
logs:
	docker logs $(CONTAINER_NAME)

# Clean up everything
clean:
	@echo "Cleaning up Docker resources..."
	@if docker ps -aq -f name=$(CONTAINER_NAME) | grep -q .; then \
		docker stop $(CONTAINER_NAME) 2>/dev/null || true; \
		docker rm $(CONTAINER_NAME) 2>/dev/null || true; \
		echo "Container removed"; \
	fi
	@if docker images -q $(IMAGE_NAME) | grep -q .; then \
		docker rmi $(IMAGE_NAME); \
		echo "Image removed"; \
	fi
	@echo "Cleanup complete!"

# Run basic tests
test: build
	@echo "Running basic functionality tests..."
	@echo "1. Testing container startup..."
	@docker run --rm --privileged $(IMAGE_NAME) echo "✓ Container starts successfully"
	@echo "2. Testing mininet installation..."
	@docker run --rm --privileged $(IMAGE_NAME) mn --version
	@echo "3. Testing ryu installation..."
	@docker run --rm --privileged $(IMAGE_NAME) ryu-manager --version
	@echo "4. Testing python..."
	@docker run --rm --privileged $(IMAGE_NAME) python3 --version
	@echo "All tests passed! ✓"