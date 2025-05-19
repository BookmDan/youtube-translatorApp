import React, { useState } from 'react';
import { 
  Container, 
  TextField, 
  Button, 
  Typography, 
  Box,
  Paper,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import axios from 'axios';

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/translate', {
        url: url
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          HomeTax Translation
        </Typography>
        <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="YouTube URL"
              variant="outlined"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter YouTube video URL"
              sx={{ mb: 2 }}
            />
            <Button 
              variant="contained" 
              color="primary" 
              type="submit"
              disabled={loading || !url}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Translate Video'}
            </Button>
          </form>
          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
          {result && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Processing Status:</Typography>
              <Typography>{result.message}</Typography>
              <Typography variant="subtitle1" sx={{ mt: 1 }}>
                {result.result && (
                  <>
                    <strong>Korean:</strong> {result.result.original}<br/>
                    <strong>English:</strong> {result.result.translation}
                  </>
                )}
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default App;
