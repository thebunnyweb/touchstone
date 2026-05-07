import { useRouter } from 'next/router'
import { useConfig } from 'nextra-theme-docs'

export default {
  logo: (
    <span style={{ fontWeight: 800, fontSize: '1.1rem', letterSpacing: '-0.02em' }}>
      ◆ touchstone
    </span>
  ),

  project: {
    link: 'https://github.com/thebunnyweb/touchstone',
  },

  docsRepositoryBase: 'https://github.com/thebunnyweb/touchstone/tree/main/docs-site',

  useNextSeoProps() {
    const { asPath } = useRouter()
    if (asPath === '/') return { titleTemplate: 'touchstone — Bias-Aware LLM Judge' }
    return { titleTemplate: '%s — touchstone' }
  },

  head() {
    const { frontMatter } = useConfig()
    return (
      <>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content={frontMatter.description || 'Bias-aware LLM judge with multi-model consensus and calibration drift detection'} />
      </>
    )
  },

  primaryHue: 262,
  primarySaturation: 60,

  sidebar: {
    defaultMenuCollapseLevel: 1,
    toggleButton: true,
  },

  toc: { backToTop: true },
  navigation: { prev: true, next: true },

  footer: {
    text: (
      <span>
        MIT {new Date().getFullYear()} ©{' '}
        <a href="https://github.com/thebunnyweb/touchstone" target="_blank">
          touchstone
        </a>
      </span>
    ),
  },

  feedback: {
    content: 'Question? Give us feedback →',
    labels: 'feedback',
  },

  editLink: {
    text: 'Edit this page on GitHub →',
  },
}
