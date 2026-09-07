// Didactic synthetic log generator. Its purpose is to TEACH the structure of a
// training corpus before moving to real data: every field is explicit, so the
// reader sees exactly the "normal language" the model will learn.

const SERVICES: &[&str] = &["auth", "api", "db", "cache", "worker"];
const METHODS: &[&str] = &["GET", "POST", "PUT", "DELETE"];
const PATHS: &[&str] = &["/login", "/logout", "/users", "/orders", "/cart", "/health", "/search"];
const CODES_OK: &[&str] = &["200", "201", "204", "301", "302"];
const CODES_ERR: &[&str] = &["400", "401", "403", "404", "500", "503"];

fn choice<'a>(xs: &[&'a str]) -> &'a str {
    xs[fastrand::usize(0..xs.len())]
}

// a NORMAL line: it obeys the service grammar
pub fn normal_line() -> String {
    // 80% INFO, 15% WARN, 5% ERROR - the typical mix of a healthy service
    let r = fastrand::f32();
    let (lvl, err) = if r < 0.80 {
        ("INFO", false)
    } else if r < 0.95 {
        ("WARN", false)
    } else {
        ("ERROR", true)
    };
    let code = if err { choice(CODES_ERR) } else { choice(CODES_OK) };
    let lat = if err {
        fastrand::u32(200..900)
    } else {
        fastrand::u32(2..250)
    };
    format!(
        "svc={} lvl={} {} {} {} {}ms",
        choice(SERVICES),
        lvl,
        choice(METHODS),
        choice(PATHS),
        code,
        lat
    )
}

// the four anomaly categories used in the evaluation
pub fn anomaly(kind: &str) -> String {
    match kind {
        // syntactic: they violate the grammar (symbols/sequences never seen)
        "garbage" => "svc=??? lvl=?? PATCH /../etc/passwd 999 -1ms".to_string(),
        "weird_method" => format!(
            "svc={} lvl=INFO TRACE /admin 200 5ms",
            choice(SERVICES)
        ),
        // semantic: valid values in an impossible combination
        "huge_lat" => format!(
            "svc={} lvl=INFO GET /health 200 89000ms",
            choice(SERVICES)
        ),
        "bad_code" => format!(
            "svc={} lvl=INFO POST /users 500 12ms",
            choice(SERVICES)
        ),
        _ => normal_line(),
    }
}

pub fn build_corpus(n: usize) -> String {
    let mut s = String::new();
    for _ in 0..n {
        s.push_str(&normal_line());
        s.push('\n');
    }
    s
}
