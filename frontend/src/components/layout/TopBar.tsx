import { ThemeToggle } from "../ThemeToggle";
import { Menu } from "lucide-react";
import type { Page } from "../../types";
export function TopBar({activePage,onMobileMenu}:{activePage:Page;onMobileMenu:()=>void}){
  return <header className={`topbar${activePage === "overview" ? " overview-topbar" : ""}`}>
    <button className="btn btn-ghost btn-icon" id="mobile-menu-btn" onClick={onMobileMenu} aria-label="Open navigation menu"><Menu size={19}/></button>
    <span className="toolbar-signature" lang="sa" title="The world is one family">वसुधैव कुटुम्बकम्</span>
    <ThemeToggle className="mobile-theme-toggle" />
  </header>;
}
