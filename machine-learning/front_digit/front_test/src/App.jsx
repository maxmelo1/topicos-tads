import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/infer_img', formData);
      setPrediction(response.data.digit);
    } catch (error) {
      console.error("Erro ao enviar imagem:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'sans-serif' }}>
      <h1>MNIST Digit Recognizer</h1>
      
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      {preview && (
        <div style={{ margin: '20px' }}>
          <img src={preview} alt="Preview" style={{ width: '140px', border: '1px solid #ccc' }} />
        </div>
      )}

      <button onClick={handleUpload} disabled={!selectedFile || loading}>
        {loading ? 'Processando...' : 'Reconhecer Dígito'}
      </button>

      {prediction !== null && (
        <div style={{ marginTop: '20px', fontSize: '2rem' }}>
          Resultado: <strong>{prediction}</strong>
        </div>
      )}
    </div>
  )
}

export default App
