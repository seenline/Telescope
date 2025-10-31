# Telescope

# README

## Instructions

### Compile and Run the System

1. **Deploy and configure AlloyDB for PostgreSQL with Docker (using the AlloyDB Omni image).**
   ```bash
   mkdir -p ./alloydb-data
   docker run -d --name alloydb \
     -e POSTGRES_PASSWORD=<YOUR_PASSWORD> \
     -v $PWD/alloydb-data:/var/lib/postgresql/data \
     -p 5432:5432 \
     --restart=always \
     google/alloydbomni:15

2. **Verify the connection and set the DSN.**

3. **Create a Python environment and install dependencies.**

4. **Build offline priority metadata with `e num.py` and `compare.py`.**

5. **Start the Telescope what-if estimator with `/HtapFormer`**

6. **Run a sample what-if call (predict latency without loading columns).**

2. **Testing Column Selection Code**



By following these steps, you should be able to compile, run, and test the system effectively.
