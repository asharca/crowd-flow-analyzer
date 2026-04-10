import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import DashboardPage from './pages/DashboardPage'
import UploadPage from './pages/UploadPage'
import VideoListPage from './pages/VideoListPage'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white shadow-sm border-b">
            <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6">
              <Link to="/" className="text-lg font-bold text-gray-800">
                Crowd Flow Analyzer
              </Link>
              <Link
                to="/"
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Upload
              </Link>
              <Link
                to="/videos"
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Videos
              </Link>
            </div>
          </nav>
          <main className="max-w-6xl mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/videos" element={<VideoListPage />} />
              <Route path="/videos/:id" element={<DashboardPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
