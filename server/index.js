import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { WebSocketServer } from 'ws';

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Backtesting endpoint
app.post('/api/backtest', async (req, res) => {
  try {
    const { symbols, allocations, startDate, endDate, strategy } = req.body;
    // Import your existing backtesting logic here
    const results = {
      initial_capital: 100000,
      final_net_worth: 120000,
      total_return: 20,
      total_trades: 50,
      trade_log_df: [],
      pnl_chart_data: []
    };
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// WebSocket for live price updates
const wss = new WebSocketServer({ server: httpServer });

wss.on('connection', (ws) => {
  console.log('Client connected');
  
  ws.on('message', (message) => {
    const data = JSON.parse(message);
    if (data.type === 'subscribe') {
      // Connect to Binance WebSocket and forward data
      const binanceWs = new WebSocket(`wss://stream.binance.com:9443/ws/${data.symbol.toLowerCase()}@trade`);
      
      binanceWs.on('message', (binanceData) => {
        ws.send(binanceData);
      });
    }
  });
});

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});