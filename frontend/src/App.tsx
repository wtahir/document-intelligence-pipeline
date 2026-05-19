import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ConfigProvider } from './context/ConfigContext'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Overview from './pages/Overview'
import PipelineRunner from './pages/PipelineRunner'
import DocumentExplorer from './pages/DocumentExplorer'
import QueryInterface from './pages/QueryInterface'
import Evaluation from './pages/Evaluation'

export default function App() {
  return (
    <ConfigProvider>
    <BrowserRouter>
      <Routes>
        {/* Landing — instant load, no API dependency */}
        <Route index element={<Landing />} />

        {/* Dashboard — full app with sidebar */}
        <Route path="dashboard" element={<Layout />}>
          <Route index          element={<Overview />} />
          <Route path="pipeline"   element={<PipelineRunner />} />
          <Route path="documents"  element={<DocumentExplorer />} />
          <Route path="query"      element={<QueryInterface />} />
          <Route path="evaluation" element={<Evaluation />} />
        </Route>
      </Routes>
    </BrowserRouter>
    </ConfigProvider>
  )
}
