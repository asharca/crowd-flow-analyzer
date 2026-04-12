import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom'
import DashboardPage from './pages/DashboardPage'
import UploadPage from './pages/UploadPage'
import VideoListPage from './pages/VideoListPage'

const queryClient = new QueryClient()

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  const { pathname } = useLocation()
  const isActive = to === '/' ? pathname === '/' : pathname.startsWith(to)

  return (
    <Link
      to={to}
      className={`text-sm px-3 py-1.5 rounded-md transition-colors ${
        isActive
          ? 'bg-blue-50 text-blue-700 font-medium'
          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
      }`}
    >
      {children}
    </Link>
  )
}

function Layout() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-1">
              <Link to="/" className="flex items-center gap-2 mr-6">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                  <svg className="w-4.5 h-4.5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
                  </svg>
                </div>
                <span className="text-base font-semibold text-gray-900 hidden sm:block">
                  CrowdFlow
                </span>
              </Link>
              <NavLink to="/">Upload</NavLink>
              <NavLink to="/videos">Videos</NavLink>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/videos" element={<VideoListPage />} />
          <Route path="/videos/:id" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout />
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
