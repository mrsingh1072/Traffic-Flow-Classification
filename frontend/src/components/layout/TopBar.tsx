import { ThemeToggle } from "../ThemeToggle";
import { Menu } from "lucide-react";
import type { HealthResponse } from "../../services/api";
import type { Page } from "../../types";
const titles: Record<Page, string> = {overview:"Overview",classifier:"Traffic Classifier",evaluation:"Model Evaluation",datasets:"Dataset Explorer",about:"About"};
export function TopBar({activePage,health,healthLoading,healthError,onMobileMenu}:{activePage:Page;health:HealthResponse|null;healthLoading:boolean;healthError:string|null;onMobileMenu:()=>void}){
  const online=!!health&&!healthError;
  const label=healthLoading&&!health?"Checking backend…":online?"Backend online":"Backend offline";
  return <header className={`topbar${activePage === "overview" ? " overview-topbar" : ""}`}>
    <button className="btn btn-ghost btn-icon" id="mobile-menu-btn" onClick={onMobileMenu} aria-label="Open navigation menu"><Menu size={19}/></button>
    {activePage === "overview" ? <span className="toolbar-signature" lang="sa" title="The world is one family">वसुधैव कुटुम्बकम्</span> : <span className="toolbar-page">{titles[activePage]}</span>}
    {activePage !== "overview" && <div className={`backend-state ${online?"online":"offline"}`} role="status" title={healthError||undefined}><span className="connection-dot"/>{label}</div>}
    <ThemeToggle className="mobile-theme-toggle" />
  </header>;
}
