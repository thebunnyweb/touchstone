import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx',
  defaultShowCopyCode: true,
})

const isProd = process.env.NODE_ENV === 'production'

export default withNextra({
  output: 'export',
  basePath: isProd ? '/touchstone' : '',
  images: { unoptimized: true },
})
