import { Github, Linkedin, Mail, MapPin } from 'lucide-react'

const placeholder = '#'

export function Footer() {
  return (
    <footer className="site-footer" id="about">
      <div className="footer-grid">
        <div><h2>PneumoVision AI</h2><p>AI-powered chest X-ray analysis for research and educational purposes.</p><p className="footer-project-note">DenseNet121-based classification and model interpretability demonstration. It is not a clinical deployment or medical diagnosis tool.</p></div>
        <div><h3>Developer</h3><p className="developer-name">Md. Mahfujur Rahman</p><p>Computer Science &amp; Engineering<br />Daffodil International University</p><p className="roles">Project Lead · AI Engineer · Full Stack Developer · UI/UX Designer · IoT System Designer</p><p className="location"><MapPin size={15} /> Daffodil International University, Dhaka, Bangladesh</p></div>
        <div><h3>Quick Links</h3><a href="#top">Home</a><a href="#analyze">Analyze X-Ray</a><a href="#model">Model</a><a href="#about">About</a></div>
        <div><h3>Connect</h3><a href={placeholder} onClick={(event) => event.preventDefault()}><Github size={16} /> GitHub <span>Coming soon</span></a><a href={placeholder} onClick={(event) => event.preventDefault()}><Linkedin size={16} /> LinkedIn <span>Coming soon</span></a><a href={placeholder} onClick={(event) => event.preventDefault()}><Mail size={16} /> Email <span>Coming soon</span></a></div>
      </div>
      <div className="footer-bottom"><span>© 2026 PneumoVision AI. All rights reserved.</span><span>Research &amp; Educational Use Only.</span></div>
    </footer>
  )
}
