;; ========================================
;; Cybersecurity Threat Knowledge Base
;; MeTTa Ontology for NL Query System
;; ========================================

;; ========================================
;; Type Declarations
;; ========================================

(: ThreatType Type)
(: Vulnerability Type)
(: AttackVector Type)
(: Mitigation Type)
(: Severity Type)
(: Software Type)
(: KillChainPhase Type)
(: AssetType Type)

;; Severity levels
(: critical Severity)
(: high Severity)
(: medium Severity)
(: low Severity)
(: informational Severity)

;; Kill chain phases
(: reconnaissance KillChainPhase)
(: weaponization KillChainPhase)
(: delivery KillChainPhase)
(: exploitation KillChainPhase)
(: installation KillChainPhase)
(: command-and-control KillChainPhase)
(: actions-on-objectives KillChainPhase)

;; ========================================
;; Threat Types (~20 threats)
;; ========================================

(: ransomware ThreatType)
(: phishing ThreatType)
(: sql-injection ThreatType)
(: xss ThreatType)
(: ddos ThreatType)
(: man-in-the-middle ThreatType)
(: zero-day ThreatType)
(: apt ThreatType)
(: brute-force ThreatType)
(: insider-threat ThreatType)
(: supply-chain-attack ThreatType)
(: credential-stuffing ThreatType)
(: buffer-overflow ThreatType)
(: privilege-escalation ThreatType)
(: dns-spoofing ThreatType)
(: cryptojacking ThreatType)
(: social-engineering ThreatType)
(: malware ThreatType)
(: rootkit ThreatType)
(: trojan ThreatType)

;; ========================================
;; Threat Severity Assignments
;; ========================================

(threat-severity ransomware critical)
(threat-severity apt critical)
(threat-severity zero-day critical)
(threat-severity supply-chain-attack critical)
(threat-severity sql-injection high)
(threat-severity xss high)
(threat-severity buffer-overflow high)
(threat-severity privilege-escalation high)
(threat-severity phishing high)
(threat-severity man-in-the-middle high)
(threat-severity rootkit high)
(threat-severity trojan high)
(threat-severity malware high)
(threat-severity ddos medium)
(threat-severity brute-force medium)
(threat-severity credential-stuffing medium)
(threat-severity dns-spoofing medium)
(threat-severity cryptojacking medium)
(threat-severity insider-threat high)
(threat-severity social-engineering high)

;; ========================================
;; Threat Descriptions
;; ========================================

(threat-description ransomware "Malware that encrypts files and demands payment for decryption keys")
(threat-description phishing "Social engineering attack using deceptive emails or messages to steal credentials")
(threat-description sql-injection "Attack that inserts malicious SQL code through unsanitized user input")
(threat-description xss "Cross-site scripting attack that injects malicious scripts into web pages")
(threat-description ddos "Distributed denial of service attack that overwhelms systems with traffic")
(threat-description man-in-the-middle "Attack that intercepts communication between two parties")
(threat-description zero-day "Exploit targeting previously unknown vulnerability with no available patch")
(threat-description apt "Advanced persistent threat - sophisticated long-term targeted attack")
(threat-description brute-force "Attack that systematically tries all possible passwords or keys")
(threat-description insider-threat "Security threat originating from within the organization")
(threat-description supply-chain-attack "Attack targeting less-secure elements in the supply chain")
(threat-description credential-stuffing "Automated injection of stolen username/password pairs")
(threat-description buffer-overflow "Attack exploiting buffer boundary errors to execute arbitrary code")
(threat-description privilege-escalation "Attack gaining higher access levels than originally authorized")
(threat-description dns-spoofing "Attack corrupting DNS cache to redirect traffic to malicious sites")
(threat-description cryptojacking "Unauthorized use of computing resources to mine cryptocurrency")
(threat-description social-engineering "Psychological manipulation to trick users into security mistakes")
(threat-description malware "Malicious software designed to damage or gain unauthorized access")
(threat-description rootkit "Stealthy malware providing continued privileged access while hiding its presence")
(threat-description trojan "Malware disguised as legitimate software to gain system access")

;; ========================================
;; Attack Vectors (~12)
;; ========================================

(: email AttackVector)
(: web-application AttackVector)
(: network AttackVector)
(: physical AttackVector)
(: usb AttackVector)
(: api AttackVector)
(: dns AttackVector)
(: wireless AttackVector)
(: cloud AttackVector)
(: firmware AttackVector)
(: social-media AttackVector)
(: supply-chain AttackVector)

;; Threat-to-attack-vector mappings
(threat-uses-vector phishing email)
(threat-uses-vector phishing social-media)
(threat-uses-vector sql-injection web-application)
(threat-uses-vector sql-injection api)
(threat-uses-vector xss web-application)
(threat-uses-vector ddos network)
(threat-uses-vector man-in-the-middle network)
(threat-uses-vector man-in-the-middle wireless)
(threat-uses-vector brute-force network)
(threat-uses-vector brute-force web-application)
(threat-uses-vector ransomware email)
(threat-uses-vector ransomware web-application)
(threat-uses-vector insider-threat physical)
(threat-uses-vector supply-chain-attack supply-chain)
(threat-uses-vector credential-stuffing web-application)
(threat-uses-vector credential-stuffing api)
(threat-uses-vector dns-spoofing dns)
(threat-uses-vector cryptojacking web-application)
(threat-uses-vector cryptojacking cloud)
(threat-uses-vector buffer-overflow api)
(threat-uses-vector rootkit firmware)
(threat-uses-vector trojan email)
(threat-uses-vector trojan usb)
(threat-uses-vector social-engineering email)
(threat-uses-vector social-engineering social-media)
(threat-uses-vector apt email)
(threat-uses-vector apt network)
(threat-uses-vector zero-day web-application)
(threat-uses-vector zero-day network)
(threat-uses-vector privilege-escalation network)
(threat-uses-vector malware email)
(threat-uses-vector malware usb)

;; ========================================
;; Kill Chain Mappings
;; ========================================

(threat-kill-chain-phase phishing delivery)
(threat-kill-chain-phase social-engineering reconnaissance)
(threat-kill-chain-phase sql-injection exploitation)
(threat-kill-chain-phase xss exploitation)
(threat-kill-chain-phase buffer-overflow exploitation)
(threat-kill-chain-phase ransomware actions-on-objectives)
(threat-kill-chain-phase malware installation)
(threat-kill-chain-phase rootkit installation)
(threat-kill-chain-phase trojan installation)
(threat-kill-chain-phase apt command-and-control)
(threat-kill-chain-phase privilege-escalation exploitation)
(threat-kill-chain-phase ddos actions-on-objectives)
(threat-kill-chain-phase cryptojacking actions-on-objectives)

;; Kill chain phase ordering (threat-precedes = can lead to)
(threat-precedes phishing ransomware)
(threat-precedes phishing credential-stuffing)
(threat-precedes phishing malware)
(threat-precedes social-engineering phishing)
(threat-precedes credential-stuffing privilege-escalation)
(threat-precedes privilege-escalation ransomware)
(threat-precedes privilege-escalation apt)
(threat-precedes sql-injection privilege-escalation)
(threat-precedes buffer-overflow privilege-escalation)
(threat-precedes xss credential-stuffing)
(threat-precedes xss phishing)
(threat-precedes trojan rootkit)
(threat-precedes trojan ransomware)
(threat-precedes malware cryptojacking)
(threat-precedes malware rootkit)
(threat-precedes zero-day privilege-escalation)
(threat-precedes supply-chain-attack malware)
(threat-precedes dns-spoofing man-in-the-middle)
(threat-precedes man-in-the-middle credential-stuffing)
(threat-precedes brute-force privilege-escalation)

;; ========================================
;; Threat Targets
;; ========================================

(threat-targets ransomware "data" "file-systems")
(threat-targets phishing "credentials" "user-accounts")
(threat-targets sql-injection "databases" "data")
(threat-targets xss "browsers" "user-sessions")
(threat-targets ddos "servers" "network-infrastructure")
(threat-targets man-in-the-middle "network-traffic" "communications")
(threat-targets brute-force "authentication-systems" "passwords")
(threat-targets insider-threat "sensitive-data" "intellectual-property")
(threat-targets cryptojacking "computing-resources" "cpu")
(threat-targets apt "critical-infrastructure" "intellectual-property")
(threat-targets supply-chain-attack "software-dependencies" "build-systems")
(threat-targets credential-stuffing "user-accounts" "authentication-systems")
(threat-targets buffer-overflow "memory" "system-processes")
(threat-targets privilege-escalation "access-controls" "system-permissions")
(threat-targets dns-spoofing "dns-infrastructure" "network-routing")
(threat-targets rootkit "operating-system" "kernel")
(threat-targets trojan "user-systems" "applications")
(threat-targets malware "systems" "data")
(threat-targets zero-day "unpatched-systems" "applications")
(threat-targets social-engineering "employees" "credentials")

;; ========================================
;; Vulnerabilities (~8 CVEs)
;; ========================================

(vulnerability "CVE-2021-44228" "Log4Shell" "Remote code execution in Apache Log4j via JNDI lookup" critical)
(vulnerability "CVE-2021-34527" "PrintNightmare" "Remote code execution in Windows Print Spooler" critical)
(vulnerability "CVE-2023-44487" "HTTP/2 Rapid Reset" "DDoS amplification via HTTP/2 stream reset" high)
(vulnerability "CVE-2021-26855" "ProxyLogon" "Server-side request forgery in Microsoft Exchange" critical)
(vulnerability "CVE-2017-0144" "EternalBlue" "Remote code execution in Windows SMB" critical)
(vulnerability "CVE-2014-0160" "Heartbleed" "Information disclosure in OpenSSL TLS heartbeat" high)
(vulnerability "CVE-2023-23397" "Outlook-Elevation" "Privilege escalation in Microsoft Outlook" critical)
(vulnerability "CVE-2019-19781" "Citrix-ADC-RCE" "Remote code execution in Citrix ADC and Gateway" critical)

;; Vulnerability-software mappings
(vulnerability-affects "CVE-2021-44228" "Apache Log4j")
(vulnerability-affects "CVE-2021-34527" "Windows Print Spooler")
(vulnerability-affects "CVE-2023-44487" "HTTP/2 implementations")
(vulnerability-affects "CVE-2021-26855" "Microsoft Exchange Server")
(vulnerability-affects "CVE-2017-0144" "Windows SMB")
(vulnerability-affects "CVE-2014-0160" "OpenSSL")
(vulnerability-affects "CVE-2023-23397" "Microsoft Outlook")
(vulnerability-affects "CVE-2019-19781" "Citrix ADC")

;; Vulnerability-threat enablement
(vulnerability-enables "CVE-2021-44228" ransomware)
(vulnerability-enables "CVE-2021-44228" apt)
(vulnerability-enables "CVE-2021-44228" cryptojacking)
(vulnerability-enables "CVE-2021-34527" privilege-escalation)
(vulnerability-enables "CVE-2021-34527" ransomware)
(vulnerability-enables "CVE-2023-44487" ddos)
(vulnerability-enables "CVE-2021-26855" apt)
(vulnerability-enables "CVE-2021-26855" ransomware)
(vulnerability-enables "CVE-2017-0144" ransomware)
(vulnerability-enables "CVE-2017-0144" malware)
(vulnerability-enables "CVE-2014-0160" man-in-the-middle)
(vulnerability-enables "CVE-2014-0160" credential-stuffing)
(vulnerability-enables "CVE-2023-23397" privilege-escalation)
(vulnerability-enables "CVE-2019-19781" ransomware)
(vulnerability-enables "CVE-2019-19781" apt)

;; ========================================
;; Mitigations (~15)
;; ========================================

;; Mitigation-to-threat mappings (flat atoms for pattern matching)
;;
;; EXTENSIBILITY: To add a new mitigation, add:
;;   (mitigation-addresses "new-mitigation-name" threat-name)
;;   (mitigation-description "new-mitigation-name" "Description text")
;; No Python code changes needed — rules automatically pick it up.

;; input-validation
(mitigation-addresses "input-validation" sql-injection)
(mitigation-addresses "input-validation" xss)
(mitigation-addresses "input-validation" buffer-overflow)

;; patch-management
(mitigation-addresses "patch-management" zero-day)
(mitigation-addresses "patch-management" ransomware)
(mitigation-addresses "patch-management" apt)

;; network-segmentation
(mitigation-addresses "network-segmentation" apt)
(mitigation-addresses "network-segmentation" ransomware)
(mitigation-addresses "network-segmentation" privilege-escalation)

;; multi-factor-auth
(mitigation-addresses "multi-factor-auth" phishing)
(mitigation-addresses "multi-factor-auth" brute-force)
(mitigation-addresses "multi-factor-auth" credential-stuffing)

;; encryption
(mitigation-addresses "encryption" man-in-the-middle)
(mitigation-addresses "encryption" insider-threat)

;; rate-limiting
(mitigation-addresses "rate-limiting" ddos)
(mitigation-addresses "rate-limiting" brute-force)
(mitigation-addresses "rate-limiting" credential-stuffing)

;; security-awareness-training
(mitigation-addresses "security-awareness-training" phishing)
(mitigation-addresses "security-awareness-training" social-engineering)
(mitigation-addresses "security-awareness-training" insider-threat)

;; web-application-firewall
(mitigation-addresses "web-application-firewall" sql-injection)
(mitigation-addresses "web-application-firewall" xss)
(mitigation-addresses "web-application-firewall" ddos)

;; endpoint-protection
(mitigation-addresses "endpoint-protection" malware)
(mitigation-addresses "endpoint-protection" ransomware)
(mitigation-addresses "endpoint-protection" rootkit)
(mitigation-addresses "endpoint-protection" trojan)

;; access-control
(mitigation-addresses "access-control" privilege-escalation)
(mitigation-addresses "access-control" insider-threat)

;; backup-strategy
(mitigation-addresses "backup-strategy" ransomware)

;; code-review
(mitigation-addresses "code-review" sql-injection)
(mitigation-addresses "code-review" xss)
(mitigation-addresses "code-review" buffer-overflow)
(mitigation-addresses "code-review" supply-chain-attack)

;; intrusion-detection
(mitigation-addresses "intrusion-detection" apt)
(mitigation-addresses "intrusion-detection" malware)
(mitigation-addresses "intrusion-detection" rootkit)

;; dns-security
(mitigation-addresses "dns-security" dns-spoofing)
(mitigation-addresses "dns-security" man-in-the-middle)

;; supply-chain-verification
(mitigation-addresses "supply-chain-verification" supply-chain-attack)
(mitigation-addresses "supply-chain-verification" trojan)

;; CVE-specific mitigations
(cve-mitigated-by "CVE-2021-44228" "patch-management")
(cve-mitigated-by "CVE-2021-44228" "input-validation")
(cve-mitigated-by "CVE-2021-44228" "web-application-firewall")
(cve-mitigated-by "CVE-2021-34527" "patch-management")
(cve-mitigated-by "CVE-2021-34527" "access-control")
(cve-mitigated-by "CVE-2023-44487" "rate-limiting")
(cve-mitigated-by "CVE-2023-44487" "web-application-firewall")
(cve-mitigated-by "CVE-2021-26855" "patch-management")
(cve-mitigated-by "CVE-2021-26855" "network-segmentation")
(cve-mitigated-by "CVE-2017-0144" "patch-management")
(cve-mitigated-by "CVE-2017-0144" "network-segmentation")
(cve-mitigated-by "CVE-2014-0160" "patch-management")
(cve-mitigated-by "CVE-2014-0160" "encryption")
(cve-mitigated-by "CVE-2023-23397" "patch-management")
(cve-mitigated-by "CVE-2019-19781" "patch-management")
(cve-mitigated-by "CVE-2019-19781" "network-segmentation")

;; Mitigation descriptions
(mitigation-description "input-validation" "Validate, sanitize, and parameterize all user inputs to prevent injection attacks")
(mitigation-description "patch-management" "Establish timely patch deployment processes for all software and systems")
(mitigation-description "network-segmentation" "Divide networks into security zones to limit blast radius of breaches")
(mitigation-description "multi-factor-auth" "Require multiple verification factors for user authentication")
(mitigation-description "encryption" "Use strong encryption (AES-256, TLS 1.3) for data protection")
(mitigation-description "rate-limiting" "Throttle requests to prevent abuse and volumetric attacks")
(mitigation-description "security-awareness-training" "Regular training on recognizing phishing, social engineering, and security hygiene")
(mitigation-description "web-application-firewall" "Filter and monitor HTTP traffic to detect and block attacks")
(mitigation-description "endpoint-protection" "Deploy EDR/XDR solutions for real-time threat detection on endpoints")
(mitigation-description "access-control" "Enforce least privilege access and regular permission reviews")
(mitigation-description "backup-strategy" "Implement 3-2-1 backup rule with regular testing of restores")
(mitigation-description "code-review" "Systematic review of source code for security vulnerabilities")
(mitigation-description "intrusion-detection" "Monitor network and system activity for signs of compromise")
(mitigation-description "dns-security" "Implement DNSSEC validation and monitor DNS query patterns")
(mitigation-description "supply-chain-verification" "Verify checksums, signatures, and provenance of dependencies")

;; ========================================
;; Reasoning Rules (Deductive)
;; ========================================
;;
;; These rules derive NEW facts from the knowledge base through composition.
;; All reasoning happens here in MeTTa — the Python side only calls rules
;; and parses results. Adding new facts above automatically flows through
;; these rules without any Python code changes.
;;

;; --- Direct Lookup Rules ---

;; Find mitigations for a threat type (via flat mitigation-addresses atoms)
(= (mitigations-for-threat $threat)
   (match &self (mitigation-addresses $name $threat) $name))

;; Find threats enabled by a CVE
(= (threats-from-cve $cve)
   (match &self (vulnerability-enables $cve $threat)
          $threat))

;; Find what software a CVE affects
(= (software-from-cve $cve)
   (match &self (vulnerability-affects $cve $software)
          $software))

;; Find CVEs for a given threat
(= (cves-enabling-threat $threat)
   (match &self (vulnerability-enables $cve $threat)
          $cve))

;; Find mitigations for a CVE
(= (mitigations-for-cve $cve)
   (match &self (cve-mitigated-by $cve $mitigation)
          $mitigation))

;; Find threats by severity level
(= (threats-by-severity $severity)
   (match &self (threat-severity $threat $severity)
          $threat))

;; Find attack vectors for a threat
(= (vectors-for-threat $threat)
   (match &self (threat-uses-vector $threat $vector)
          $vector))

;; Find all threats using a specific attack vector
(= (threats-using-vector $vector)
   (match &self (threat-uses-vector $threat $vector)
          $threat))

;; Find targets for a threat (returns each target individually)
(= (targets-for-threat $threat)
   (match &self (threat-targets $threat $primary $secondary)
          $primary))
(= (targets-for-threat $threat)
   (match &self (threat-targets $threat $primary $secondary)
          $secondary))

;; --- Kill Chain Rules ---

;; Kill chain: direct next step (1-hop)
(= (kill-chain-next $threat)
   (match &self (threat-precedes $threat $next)
          $next))

;; Kill chain: direct previous step
(= (kill-chain-previous $threat)
   (match &self (threat-precedes $prev $threat)
          $prev))

;; Transitive kill chain: 1-hop reachability
(= (kill-chain-reachable $threat)
   (match &self (threat-precedes $threat $next)
          $next))

;; Transitive kill chain: 2-hop reachability (deductive — derives paths not in the KB)
(= (kill-chain-reachable $threat)
   (match &self (threat-precedes $threat $mid)
          (match &self (threat-precedes $mid $end)
                 $end)))

;; Transitive kill chain: 3-hop reachability
(= (kill-chain-reachable $threat)
   (match &self (threat-precedes $threat $a)
          (match &self (threat-precedes $a $b)
                 (match &self (threat-precedes $b $c)
                        $c))))

;; --- Multi-Hop Deductive Rules ---

;; 2-hop: software -> CVE -> CVE-specific mitigation
(= (mitigations-for-software $software)
   (match &self (vulnerability-affects $cve $software)
          (match &self (cve-mitigated-by $cve $mitigation)
                 $mitigation)))

;; 2-hop: software -> CVE -> threats
(= (threats-for-software $software)
   (match &self (vulnerability-affects $cve $software)
          (match &self (vulnerability-enables $cve $threat)
                 $threat)))

;; 3-hop: comprehensive mitigations for software
;; Combines CVE-specific mitigations with threat-type mitigations.
;; Definition 1: CVE-specific mitigations (delegates to mitigations-for-software)
(= (all-mitigations-for-software $software)
   (mitigations-for-software $software))
;; Definition 2: 3-hop threat-type mitigations (software -> CVE -> threat -> mitigation)
(= (all-mitigations-for-software $software)
   (match &self (vulnerability-affects $cve $software)
          (match &self (vulnerability-enables $cve $threat)
                 (match &self (mitigation-addresses $mit $threat)
                        $mit))))

;; 2-hop: comprehensive mitigations for a CVE (both CVE-specific and threat-type)
;; Definition 1: CVE-specific mitigations
(= (comprehensive-cve-mitigations $cve)
   (match &self (cve-mitigated-by $cve $mitigation)
          $mitigation))
;; Definition 2: threat-type mitigations for threats enabled by the CVE
(= (comprehensive-cve-mitigations $cve)
   (match &self (vulnerability-enables $cve $threat)
          (match &self (mitigation-addresses $mitigation $threat)
                 $mitigation)))

;; --- Inference Rules ---

;; Infer if a threat has critical CVEs associated with it
(= (threat-has-critical-cve $threat)
   (match &self (vulnerability-enables $cve $threat)
          (match &self (vulnerability $cve $name $desc critical)
                 $cve)))

;; Find threats targeting a specific asset type
(= (threats-targeting $asset)
   (match &self (threat-targets $threat $primary $secondary)
          (if (== $primary $asset)
              $threat
              (if (== $secondary $asset)
                  $threat
                  (empty)))))

;; ========================================
;; End of Cybersecurity Threat Knowledge Base
;; ========================================
;;
;; EXTENSIBILITY GUIDE:
;; To add a new threat:
;;   (: new-threat ThreatType)
;;   (threat-severity new-threat high)
;;   (threat-description new-threat "Description of the new threat")
;;   (threat-uses-vector new-threat email)
;;   (threat-targets new-threat "target1" "target2")
;;   (threat-precedes new-threat ransomware)  ;; optional: kill chain link
;;
;; To add a new CVE:
;;   (vulnerability "CVE-YYYY-NNNNN" "Name" "Description" critical)
;;   (vulnerability-affects "CVE-YYYY-NNNNN" "Software Name")
;;   (vulnerability-enables "CVE-YYYY-NNNNN" threat-name)
;;   (cve-mitigated-by "CVE-YYYY-NNNNN" "mitigation-name")
;;
;; To add a new mitigation:
;;   (mitigation-addresses "new-mitigation" threat-name)
;;   (mitigation-description "new-mitigation" "Description")
;;
;; All existing rules automatically work with new data — no Python changes needed.
;; ========================================
