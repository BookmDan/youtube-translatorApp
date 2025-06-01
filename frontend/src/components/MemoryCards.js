import React, { useState } from 'react';
import './MemoryCards.css';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

const MemoryCards = ({ videoId, onClose }) => {
    const [memoryCards, setMemoryCards] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [flipped, setFlipped] = useState({});
    const [fetched, setFetched] = useState(false);

    const fetchMemoryCards = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetch(`http://localhost:8080/memory-cards/${videoId}`);
            const data = await response.json();
            if (data.status === 'success') {
                setMemoryCards(data.memory_cards);
                setFetched(true);
            } else {
                setError('Failed to load memory cards');
            }
        } catch (err) {
            setError('Error loading memory cards');
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    React.useEffect(() => {
        if (videoId && !fetched) {
            fetchMemoryCards();
        }
        // eslint-disable-next-line
    }, [videoId]);

    const handleFlip = (idx) => {
        setFlipped((prev) => ({ ...prev, [idx]: !prev[idx] }));
    };

    return (
        <div className="memory-cards-overlay">
            <button className="back-btn" onClick={onClose}>
                <ArrowBackIcon /> Back
            </button>
            <h2 className="memory-cards-title">Memory Cards</h2>
            {loading && <div className="memory-cards-loading">Loading...</div>}
            {error && <div className="error-message">{error}</div>}
            <div className="cards-grid-full">
                {memoryCards.map((card, index) => (
                    <div
                        key={index}
                        className={`memory-card-full ${flipped[index] ? 'flipped' : ''}`}
                        onClick={() => handleFlip(index)}
                    >
                        <div className="card-inner-full">
                            <div className="card-front-full">
                                <h3>{card.phrase}</h3>
                                <p>Frequency: {card.frequency}</p>
                            </div>
                            <div className="card-back-full">
                                <h3>{card.translation}</h3>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MemoryCards; 