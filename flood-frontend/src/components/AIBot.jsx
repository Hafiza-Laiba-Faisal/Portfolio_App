import { useState } from "react";
import "../styles/aibot.css";
import { orchestrateAction } from "../services/api";

export default function AIBot() {
  const [userMessage, setUserMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [showRobot, setShowRobot] = useState(true);
  const [loading, setLoading] = useState(false);

  const handleKeyPress = async (e) => {
    if (e.key === "Enter" && userMessage.trim()) {
      const newMessage = { sender: "user", text: userMessage };
      setChatHistory((prev) => [...prev, newMessage]);
      setUserMessage("");
      setShowRobot(false);
      setLoading(true);

      // 🔗 Call demo API
      const response = await orchestrateAction(userMessage);

      const botMessage = {
        sender: "bot",
        text: response.error
          ? response.error
          : JSON.stringify(response, null, 2)
      };

      setChatHistory((prev) => [...prev, botMessage]);
      setLoading(false);
    }
  };

  return (
    <div className="aibot">
      <div className="chat">
        <div className="chat-window">
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              className={`chat-message ${msg.sender === "user" ? "user" : "bot"}`}
            >
              {msg.text}
            </div>
          ))}
          {loading && <div className="chat-message bot">⏳ Thinking...</div>}
        </div>

        {showRobot && (
          <div className="robot" id="robot">
            <video muted autoPlay loop playsInline src="/videos/robot.mp4"></video>
            <h2 className="text-color-effect">Ask AI Anything</h2>
            <p>Flood forecasting, evacuation, reconstruction & more 🚨</p>
          </div>
        )}
      </div>

      <div className="input">
        <input
          type="text"
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message & press Enter"
        />
        <div className="videodiv">
          <video muted autoPlay loop src="/videos/ezgif-73b7f6ac862004.mp4"></video>
        </div>
      </div>
    </div>
  );
}
