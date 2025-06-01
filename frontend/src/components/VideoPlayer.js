const formatTimestamp = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

{translations.map((item, index) => (
  <div key={index} className="translation-item">
    <div className="timestamp">
      {formatTimestamp(item.start)}
    </div>
    <div className="translation-content">
      <div className="original-text">{item.original}</div>
      <div className="translated-text">{item.translated}</div>
    </div>
  </div>
))} 