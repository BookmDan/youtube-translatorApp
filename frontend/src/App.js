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
  FormControl,
  Tooltip
} from '@mui/material';
import YouTubeIcon from '@mui/icons-material/YouTube';
import TranslateIcon from '@mui/icons-material/Translate';
import GitHubIcon from '@mui/icons-material/GitHub';
import SummarizeIcon from '@mui/icons-material/Summarize';
import StyleIcon from '@mui/icons-material/Style';
import axios from 'axios';
import './App.css';
import VideoPlayer from './components/VideoPlayer';
import MemoryCards from './components/MemoryCards';

const TRANSLATION_API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8080') + '/translate';
const TRANSCRIPT_API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8080') + '/read-transcript';
const SUMMARIZE_API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8080') + '/summarize-transcript';

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
  const [lastTranslatedUrl, setLastTranslatedUrl] = useState('');
  const [lastTranscriptUrl, setLastTranscriptUrl] = useState('');
  const [showTranslation, setShowTranslation] = useState(false);
  const [koreanTranscript, setKoreanTranscript] = useState('');
  const [englishTranscript, setEnglishTranscript] = useState('');
  const [showEnglish, setShowEnglish] = useState(false);
  const [videoId, setVideoId] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [translationStatus, setTranslationStatus] = useState(null);
  const [translationProgress, setTranslationProgress] = useState(0);
  const [translationMethod, setTranslationMethod] = useState(null);
  const [sourceLanguage, setSourceLanguage] = useState(null);
  const [translations, setTranslations] = useState([]);
  const [subtitleCount, setSubtitleCount] = useState(0);
  const [showMemoryCards, setShowMemoryCards] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    
    // Clear transcript if we're translating a different video
    if (url !== lastTranslatedUrl) {
      setTranscript(null);
      setSummary(null);
    }

    if (!url) {
      setError('Please enter a YouTube URL.');
      setLoading(false);
      return;
    }

    try {
      // Extract video ID from URL
      const videoIdMatch = url.match(/(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/);
      if (videoIdMatch) {
        setVideoId(videoIdMatch[1]);
      }

      const response = await axios.post(TRANSLATION_API_URL, {
        url: url,
        start_time: 0,
        end_time: null,
        target_lang: targetLang
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      setResult(response.data);
      setLastTranslatedUrl(url); // Track which URL was translated
    } catch (err) {
      console.error('Translation error:', err);
      setError(
        err.response?.data?.detail ||
        'Failed to translate. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleGetTranscript = async () => {
    setLoading(true);
    setError('');
    setTranscript('');
    setKoreanTranscript('');
    setEnglishTranscript('');
    setShowTranslation(false);
    setShowEnglish(false);

    if (!url) {
      setError('Please enter a YouTube URL.');
      setLoading(false);
      return;
    }

    try {
      // Extract video ID from URL
      const videoIdMatch = url.match(/(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/);
      if (videoIdMatch) {
        setVideoId(videoIdMatch[1]);
      }

      const response = await fetch(TRANSCRIPT_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();

      if (response.ok) {
        console.log("Transcript data:", data); // Debug log
        setKoreanTranscript(data.korean_transcript || '');
        setEnglishTranscript(data.english_transcript || '');
        setTranscript(data.korean_transcript || ''); // Default to Korean
        setShowTranslation(false);
      } else {
        setError(data.detail || 'Failed to get transcript');
      }
    } catch (err) {
      console.error('Transcript error:', err);
      setError('Error connecting to server. Please make sure the backend server is running.');
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
    setLoading(true);
    
    try {
      const response = await fetch(SUMMARIZE_API_URL, {
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

  const toggleLanguage = () => {
    setShowEnglish(!showEnglish);
    setTranscript(showEnglish ? koreanTranscript : englishTranscript);
  };

  const selectedLangLabel = LANGUAGES.find(l => l.code === targetLang)?.label || 'English';

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f6f8fa' }}>
      {/* Memory Cards Fullscreen Overlay */}
      {showMemoryCards && videoId && (
        <MemoryCards videoId={videoId} onClose={() => setShowMemoryCards(false)} />
      )}
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
      <Container maxWidth={result && transcript ? "lg" : "sm"} sx={{ mt: 6, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
          <Box display="flex" alignItems="center" mb={2}>
            <YouTubeIcon color="error" sx={{ fontSize: 40, mr: 2 }} />
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h5" fontWeight={700}>
                Translate Korean YouTube Videos
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Enter a YouTube URL to translate Korean subtitles to your selected language. The tool will extract the subtitles, translate them using a serverless function with PyTorch and the Helsinki-NLP model, and provide a side-by-side view.
              </Typography>
            </Box>
            {videoId && (
              <Tooltip title="Show top 20 most frequent phrases as memory cards">
                <Button
                  variant="outlined"
                  color="primary"
                  size="medium"
                  startIcon={<StyleIcon />}
                  sx={{
                    ml: 2,
                    borderRadius: 2,
                    fontWeight: 600,
                    fontSize: 15,
                    px: 2.5,
                    py: 0.5,
                    boxShadow: 0,
                    textTransform: 'none',
                    height: 40,
                    minWidth: 0
                  }}
                  onClick={() => setShowMemoryCards(true)}
                >
                  Memory Cards
                </Button>
              </Tooltip>
            )}
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
          
          {/* Side by Side Layout for Translation and Transcript */}
          {result && transcript && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ textAlign: 'center', mb: 3 }}>
                📊 Translation & Transcript Comparison
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3, '@media (max-width: 900px)': { gridTemplateColumns: '1fr', gap: 2 } }}>
                {/* Translation Section */}
                <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#f0f7ff', border: '2px solid #e3f2fd' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1565c0', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TranslateIcon />
                    Translation Result
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, color: '#666' }}>
                    {result.message}
                  </Typography>
                  {Array.isArray(result.result) && result.result.length > 0 && (
                    <List sx={{ maxHeight: '500px', overflow: 'auto' }}>
                      {result.result.map((line, idx) => (
                        <React.Fragment key={idx}>
                          <ListItem alignItems="flex-start" sx={{ pb: 1, px: 1 }}>
                            <ListItemText
                              primary={
                                <>
                                  <Box mb={0.5} sx={{ fontSize: '0.9rem' }}>
                                    <strong>Korean:</strong> <span>{line.original}</span>
                                  </Box>
                                  <Box sx={{ fontSize: '0.9rem', color: '#1565c0' }}>
                                    <strong>English:</strong> <span>{line.translated || line.translation}</span>
                                  </Box>
                                </>
                              }
                              secondary={`⏱️ ${line.start}s • ⏳ ${line.duration}s`}
                            />
                          </ListItem>
                          {idx < result.result.length - 1 && <Divider />}
                        </React.Fragment>
                      ))}
                    </List>
                  )}
                </Paper>

                {/* Transcript Section */}
                <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#f7f7f7', border: '2px solid #e0e0e0' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#424242', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
                    📝 Full Transcript
                  </Typography>
                  <div className="transcript-section">
                    <div className="transcript-controls">
                      <button 
                        onClick={() => setShowEnglish(false)} 
                        className={!showEnglish ? 'active' : ''}
                      >
                        Korean Transcript
                      </button>
                      <button 
                        onClick={() => setShowEnglish(true)} 
                        className={showEnglish ? 'active' : ''}
                      >
                        English Transcript
                      </button>
                    </div>
                    <div className="transcript-content">
                      {showEnglish ? englishTranscript : koreanTranscript}
                    </div>
                  </div>
                </Paper>
              </Box>
            </Box>
          )}

          {/* Loading state when translation is loading but we have old transcript */}
          {loading && result && transcript && url !== lastTranscriptUrl && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ textAlign: 'center', mb: 3 }}>
                📊 Translation & Transcript Comparison
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3, '@media (max-width: 900px)': { gridTemplateColumns: '1fr', gap: 2 } }}>
                {/* Translation Section */}
                <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#fff3cd', border: '2px solid #ffc107' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#856404', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TranslateIcon />
                    Translating New Video...
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, color: '#666' }}>
                    Processing translation for the new video URL
                  </Typography>
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body1" sx={{ color: '#856404' }}>
                      ⏳ Translating...
                    </Typography>
                  </Box>
                </Paper>

                {/* Transcript Section - Old transcript warning */}
                <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#ffe0e6', border: '2px solid #ffb3ba' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#721c24', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
                    ⚠️ Previous Transcript
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, color: '#721c24' }}>
                    This transcript is from the previous video. Click "Get Transcript" for the new video.
                  </Typography>
                  <Box sx={{ maxHeight: '400px', overflow: 'auto', fontSize: '0.9rem', lineHeight: 1.6, opacity: 0.7 }}>
                    <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0, fontFamily: 'inherit' }}>{transcript}</pre>
                  </Box>
                </Paper>
              </Box>
            </Box>
          )}

          {/* Individual Translation Section (when no transcript) */}
          {result && !transcript && (
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
                                  <strong>English:</strong> <span>{line.translated || line.translation}</span>
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

          {/* Individual Transcript Section (when no translation) */}
          {transcript && !result && (
            <Box sx={{ mt: 3 }}>
              <Paper elevation={2} sx={{ p: 3, borderRadius: 2, bgcolor: '#f7f7f7' }}>
                <Typography variant="h6" gutterBottom sx={{ color: '#424242', fontWeight: 700 }}>
                  📝 Full Transcript
                </Typography>
                <div className="transcript-section">
                  <div className="transcript-controls">
                    <button 
                      onClick={() => setShowEnglish(false)} 
                      className={!showEnglish ? 'active' : ''}
                    >
                      Korean Transcript
                    </button>
                    <button 
                      onClick={() => setShowEnglish(true)} 
                      className={showEnglish ? 'active' : ''}
                    >
                      English Transcript
                    </button>
                  </div>
                  <div className="transcript-content">
                    {showEnglish ? englishTranscript : koreanTranscript}
                  </div>
                </div>
              </Paper>
            </Box>
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
