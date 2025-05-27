import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  TextField,
  Button,
  Box,
  Alert,
  InputAdornment,
  Link,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  MenuItem,
  Select,
  InputLabel,
  FormControl
} from '@mui/material';
import YouTubeIcon from '@mui/icons-material/YouTube';
import TranslateIcon from '@mui/icons-material/Translate';
import GitHubIcon from '@mui/icons-material/GitHub';
import SummarizeIcon from '@mui/icons-material/Summarize';
import axios from 'axios';

const TRANSLATION_API_URL = 'http://localhost:8000/translate'; // Update if needed

const LANGUAGES = [
  { code: 'eng_Latn', label: 'English' },
  { code: 'spa_Latn', label: 'Spanish' },
  { code: 'fra_Latn', label: 'French' },
  { code: 'zho_Hans', label: 'Chinese (Simplified)' },
  { code: 'arb_Arab', label: 'Arabic' },
  { code: 'tl_Latn', label: 'Tagalog' },
  { code: 'swe_Latn', label: 'Swedish' },
];

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [targetLang, setTargetLang] = useState('eng_Latn');
  const [transcript, setTranscript] = useState(null);
  const [summary, setSummary] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setTranscript(null);
    setSummary(null);

    if (!url) {
      setError('Please enter a YouTube URL.');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(TRANSLATION_API_URL, {
        url: url,
        start_time: 0,
        end_time: null,
        target_lang: targetLang
      });
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Translation service URL not configured. Please set the TRANSLATION_LAMBDA_URL environment variable.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleGetTranscript = async () => {
    setError("");
    setTranscript("");
    setSummary("");
    if (!url) {
      setError("Please enter a YouTube URL.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/read-transcript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      if (response.ok && data.status === "success") {
        setTranscript(data.transcript);
      } else {
        setError(data.detail || data.message || "Failed to load transcript.");
      }
    } catch (err) {
      setError("Failed to load transcript.");
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!transcript) {
      setError("Please get the transcript first before summarizing.");
      return;
    }
    
    setError("");
    setSummary("");
    setLoading(true);
    
    try {
      const response = await fetch("http://localhost:8000/summarize-transcript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          transcript: transcript,
          url: url 
        }),
      });
      const data = await response.json();
      if (response.ok && data.status === "success") {
        setSummary(data.summary);
      } else {
        setError(data.detail || data.message || "Failed to summarize transcript.");
      }
    } catch (err) {
      setError("Failed to summarize transcript.");
    } finally {
      setLoading(false);
    }
  };

  const selectedLangLabel = LANGUAGES.find(l => l.code === targetLang)?.label || 'English';

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f6f8fa' }}>
      {/* Header */}
      <AppBar position="static" sx={{ bgcolor: '#6c47ff' }}>
        <Toolbar>
          <TranslateIcon sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            YouTube Korean Translator
          </Typography>
          <IconButton
            color="inherit"
            component={Link}
            href="https://github.com/BookmDan/youtube-korean-english-translator"
            target="_blank"
            rel="noopener"
            aria-label="GitHub"
          >
            <GitHubIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main Card */}
      <Container maxWidth="sm" sx={{ mt: 6, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
          <Box display="flex" alignItems="center" mb={2}>
            <YouTubeIcon color="error" sx={{ fontSize: 40, mr: 2 }} />
            <Box>
              <Typography variant="h5" fontWeight={700}>
                Translate Korean YouTube Videos
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Enter a YouTube URL to translate Korean subtitles to your selected language. The tool will extract the subtitles, translate them using a serverless function with PyTorch and the Helsinki-NLP model, and provide a side-by-side view.
              </Typography>
            </Box>
          </Box>
          <form onSubmit={handleSubmit}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="lang-select-label">Output Language</InputLabel>
              <Select
                labelId="lang-select-label"
                value={targetLang}
                label="Output Language"
                onChange={(e) => setTargetLang(e.target.value)}
              >
                {LANGUAGES.map(lang => (
                  <MenuItem key={lang.code} value={lang.code}>{lang.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              fullWidth
              label="YouTube URL"
              variant="outlined"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              sx={{ mb: 3 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <YouTubeIcon color="error" />
                  </InputAdornment>
                ),
              }}
            />
            <Box sx={{ mt: 2, display: 'flex', gap: 1, flexDirection: 'column' }}>
              <Button
                variant="contained"
                color="primary"
                type="submit"
                fullWidth
                size="large"
                sx={{
                  bgcolor: '#6c47ff',
                  color: '#fff',
                  fontWeight: 700,
                  py: 1.5,
                  borderRadius: 2,
                  boxShadow: 2,
                  '&:hover': { bgcolor: '#5936d9' }
                }}
                startIcon={<TranslateIcon />}
                disabled={loading}
              >
                {loading ? 'Translating...' : 'Translate Video'}
              </Button>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  onClick={handleGetTranscript}
                  disabled={loading}
                  fullWidth
                  sx={{ py: 1.5 }}
                >
                  {loading ? 'Loading...' : 'Get Transcript'}
                </Button>
                <Button
                  variant="outlined"
                  onClick={handleSummarize}
                  disabled={loading || !transcript}
                  fullWidth
                  sx={{ 
                    py: 1.5,
                    bgcolor: transcript ? '#fff3cd' : 'inherit',
                    borderColor: transcript ? '#ffc107' : 'inherit',
                    color: transcript ? '#856404' : 'inherit',
                    '&:hover': { 
                      bgcolor: transcript ? '#ffecb5' : 'inherit',
                      borderColor: transcript ? '#ffca2c' : 'inherit'
                    }
                  }}
                  startIcon={<SummarizeIcon />}
                >
                  {loading ? 'Summarizing...' : 'Summarize'}
                </Button>
              </Box>
            </Box>
          </form>
          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              <strong>Error</strong>
              <br />
              {error}
            </Alert>
          )}
          {result && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Translation Result
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {result.message}
              </Typography>
              {Array.isArray(result.result) && result.result.length > 0 && (
                <Paper elevation={1} sx={{ p: 2, borderRadius: 2, bgcolor: '#f9f9fb' }}>
                  <List>
                    {result.result.map((line, idx) => (
                      <React.Fragment key={idx}>
                        <ListItem alignItems="flex-start" sx={{ pb: 1 }}>
                          <ListItemText
                            primary={
                              <>
                                <Box mb={0.5}>
                                  <strong>Korean:</strong> <span>{line.original}</span>
                                </Box>
                                <Box>
                                  <strong>{selectedLangLabel}:</strong> <span>{line.translation}</span>
                                </Box>
                              </>
                            }
                            secondary={`Start: ${line.start}, Duration: ${line.duration}`}
                          />
                        </ListItem>
                        {idx < result.result.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                </Paper>
              )}
            </Box>
          )}
          {/* Transcript Section */}
          {transcript && (
            <div style={{ marginTop: 24, padding: 16, background: "#f7f7f7", borderRadius: 8 }}>
              <h3>Transcript</h3>
              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{transcript}</pre>
            </div>
          )}
          
          {/* Summary Section */}
          {summary && (
            <Box sx={{ mt: 3 }}>
              <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#e8f5e8' }}>
                <Typography variant="h6" gutterBottom sx={{ color: '#2e7d32', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SummarizeIcon />
                  AI Summary
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    whiteSpace: 'pre-wrap', 
                    lineHeight: 1.6,
                    color: '#1b5e20'
                  }}
                >
                  {summary}
                </Typography>
              </Paper>
            </Box>
          )}
        </Paper>
      </Container>
      {/* Footer */}
      <Box sx={{ textAlign: 'center', color: 'text.secondary', py: 3, fontSize: 14 }}>
        © {new Date().getFullYear()} YouTube Korean Translator | Built with PyTorch, HuggingFace Transformers, and Material-UI
      </Box>
    </Box>
  );
}

export default App;
