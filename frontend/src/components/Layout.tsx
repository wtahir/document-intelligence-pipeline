import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import DemoBanner from './DemoBanner'

export default function Layout() {
  return (
    <div className="flex min-h-screen font-sans">
      <Sidebar />
      <div className="flex flex-col flex-1">
        <DemoBanner />
        <main className="flex-1 overflow-y-auto bg-surface-900">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
