#!/bin/bash

# Create maps directory
MAPS_DIR=$(rospack find hybrid_navigation)/maps
mkdir -p $MAPS_DIR
chmod 777 $MAPS_DIR
echo "Created maps directory at $MAPS_DIR with full permissions" 