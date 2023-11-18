# Transformer for Translation
 Transformer for Translation is a repository for building powerful neural machine translation models using state-of-the-art transformer architectures, enabling seamless cross-lingual communication and breaking down language barriers.
# Architecture of Transfomer
 ![transformerArch](https://github.com/bhavinpatil/Transformer-for-Translation/assets/59190105/016e86d9-b6ce-4fd2-9312-003dead4b477)

## Running Jupyter Notebook Remotely

To run Jupyter Notebook on a remote server without opening a browser, follow these steps:

1. Start Jupyter Notebook on the remote server with the following command:

    ```bash
    jupyter notebook --no-browser --port=8080
    ```

2. Set up an SSH tunnel from your local machine to the remote server:

    ```bash
    ssh -L 8080:localhost:8080 <REMOTE_USER>@<REMOTE_HOST>
    ```

    Replace `<REMOTE_USER>` with your remote username and `<REMOTE_HOST>` with the address of your remote server.

3. Open your local browser and go to [http://localhost:8080/](http://localhost:8080/) to access the Jupyter Notebook running on the remote server.

Make sure to replace `<REMOTE_USER>` and `<REMOTE_HOST>` with your actual remote server credentials.
