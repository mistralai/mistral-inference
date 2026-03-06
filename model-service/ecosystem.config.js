module.exports = {
  apps: [
    {
      name: 'reentry-model-service',
      script: 'server.js',
      instances: 1,
      exec_mode: 'fork',
      env: {
        // server.js loads dotenv; these are optional overrides
        PORT: process.env.PORT || 7071,
      },
      watch: false,
    },
  ],
};

