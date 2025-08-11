const express = require('express');
const path = require('path');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, 'uga-ugas')));

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'uga-ugas', 'ugas.html'));
});

app.get('/color', (req, res) => {
  res.sendFile(path.join(__dirname, 'uga-ugas/color', 'color.html'));
})

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});