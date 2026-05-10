import { createContext, useContext, useEffect, useState } from 'react'

interface AppConfig {
  demo_mode: boolean
  version: string
}

const ConfigContext = createContext<AppConfig>({ demo_mode: false, version: '' })

export function ConfigProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<AppConfig>({ demo_mode: false, version: '' })

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(setConfig)
      .catch(() => {}) // silently fail if backend not yet ready
  }, [])

  return <ConfigContext.Provider value={config}>{children}</ConfigContext.Provider>
}

export function useConfig() {
  return useContext(ConfigContext)
}
