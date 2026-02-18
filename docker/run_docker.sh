#!/bin/bash
# ManipBench Docker launcher
#
# Usage:
#   bash docker/run_docker.sh              # Build & run
#   bash docker/run_docker.sh -r           # Force rebuild
#   bash docker/run_docker.sh -v           # Verbose
set -e

DOCKER_IMAGE_NAME='manip_bench'
DOCKER_VERSION_TAG='latest'
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKDIR="/workspaces/manip_bench"
FORCE_REBUILD=false

while getopts ":hrRv" OPTION; do
    case $OPTION in
        r) FORCE_REBUILD=true ;;
        R) FORCE_REBUILD=true; NO_CACHE="--no-cache" ;;
        v) set -x ;;
        h)
            echo "Usage: $0 [-r] [-R] [-v] [-h]"
            echo "  -r  Force rebuild"
            echo "  -R  Force rebuild without cache"
            echo "  -v  Verbose"
            exit 0
            ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

echo "ManipBench Docker — image: $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG"

if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2>/dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Image exists. Use -r to rebuild."
else
    docker build --pull $NO_CACHE \
        --build-arg WORKDIR="${WORKDIR}" \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $SCRIPT_DIR/Dockerfile \
        "$PROJECT_DIR"
fi

if [ "$(docker ps -a --quiet --filter status=exited --filter name=$DOCKER_IMAGE_NAME)" ]; then
    docker rm $DOCKER_IMAGE_NAME > /dev/null
fi

xhost +local:docker > /dev/null 2>&1 || true

docker run \
    --name "$DOCKER_IMAGE_NAME" \
    --privileged \
    --ipc=host --net=host \
    --runtime=nvidia --gpus=all \
    -v "$PROJECT_DIR/manip_bench:${WORKDIR}/manip_bench" \
    -v "$PROJECT_DIR/assets:${WORKDIR}/assets" \
    -v "$PROJECT_DIR/configs:${WORKDIR}/configs" \
    -v "$PROJECT_DIR/tools:${WORKDIR}/tools" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env DISPLAY \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --interactive --rm --tty \
    ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} "${@}"
