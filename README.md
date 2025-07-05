# rutespractic.github.io
<!DOCTYPE html>
<html lang="ca">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pujar Ruta d'Examen</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 2px dashed #3498db;
            border-radius: 5px;
        }
        input[type="submit"] {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        input[type="submit"]:hover {
            background-color: #2980b9;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Puja el teu fitxer KML</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <input type="file" id="kmlFile" name="kml_file" accept=".kml" required>
            </div>
            <div class="form-group">
                <input type="submit" value="Enviar Ruta">
            </div>
        </form>
        <div id="statusMessage" class="status"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('kmlFile');
            const statusDiv = document.getElementById('statusMessage');
            
            if (!fileInput.files.length) {
                showStatus('Si us plau, selecciona un fitxer KML', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('kml_file', fileInput.files[0]);

            try {
                // Reemplaza con tu URL de webhook n8n
                const response = await fetch('TU_ENLLAÇ_DE_N8N_WEBHOOK', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    showStatus('Ruta pujada correctament!', 'success');
                    fileInput.value = ''; // Limpiar el input
                } else {
                    throw new Error('Error en el servidor');
                }
            } catch (error) {
                console.error('Error:', error);
                showStatus(`Error en pujar la ruta: ${error.message}`, 'error');
            }
        });

        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + type;
            statusDiv.style.display = 'block';
            
            // Ocultar después de 5 segundos
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
