import chalk from 'chalk';

function ts() {
  return chalk.gray(new Date().toISOString().slice(11, 19));
}

export const log = {
  info:    (msg) => console.log(`${ts()} ${chalk.blue('INFO')}  ${msg}`),
  success: (msg) => console.log(`${ts()} ${chalk.green('OK')}    ${msg}`),
  warn:    (msg) => console.log(`${ts()} ${chalk.yellow('WARN')}  ${msg}`),
  error:   (msg) => console.log(`${ts()} ${chalk.red('ERR')}   ${msg}`),
  phase:   (msg) => console.log(`\n${ts()} ${chalk.cyan('━━━')} ${chalk.bold(msg)}\n`),
  stat:    (label, value) => console.log(`${ts()} ${chalk.magenta('STAT')}  ${label}: ${chalk.bold(value)}`),
};
