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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

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
