import React from 'react';
import { createRoot } from 'react-dom/client';
import { MantineProvider, createTheme } from '@mantine/core';
import '@mantine/core/styles.css';
import './style.css';
import App from './App.jsx';

const theme = createTheme({
  primaryColor: 'violet',
  fontFamily: 'Segoe UI, system-ui, sans-serif',
  defaultRadius: 'md',
  colors: { violet: ['#f3f0fa','#e4def4','#ccbee7','#b39ed9','#9981cb','#8166bd','#6d53ae','#5a4295','#49337b','#37255f'] },
});
createRoot(document.getElementById('root')).render(
  <MantineProvider theme={theme} defaultColorScheme="light" getStyleNonce={() => window.PALM_BOOT.nonce}>
    <App />
  </MantineProvider>,
);
