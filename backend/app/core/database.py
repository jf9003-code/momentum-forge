"""
D5 ROBUST MASTER - Database Module
SQLite database with comprehensive data persistence

FIXES APPLIED:
- Top 5 strategies now ACCUMULATE instead of being overwritten
- Unique batch_id tracking for each save operation
- Proper foreign key handling
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
import uuid
import logging

logger = logging.getLogger(__name__)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'robustmark.db')

# Ensure data directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize all database tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # ============================================
    # VALIDATIONS TABLE
    # ============================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_id TEXT UNIQUE NOT NULL,
            strategy_name TEXT,
            strategy_code TEXT,
            symbol TEXT,
            start_date TEXT,
            end_date TEXT,
            initial_capital REAL,
            position_size REAL,
            commission REAL,
            slippage REAL,
            spread REAL,
            created_at TEXT,

            -- Results
            final_score REAL,
            grade TEXT,
            decision TEXT,
            confidence_level TEXT,
            risk_level TEXT,

            -- Metrics
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            total_costs REAL,
            profit_factor REAL,
            annual_return REAL,

            -- Phase scores
            phase1_score REAL,
            phase1_passed INTEGER,
            phase2_score REAL,
            phase2_passed INTEGER,
            phase3_score REAL,
            phase3_passed INTEGER,
            phase4_score REAL,
            phase4_passed INTEGER,
            phase5_score REAL,
            phase5_passed INTEGER,
            phase6_score REAL,
            phase6_passed INTEGER,
            phase7_score REAL,
            phase7_passed INTEGER,

            -- JSON data
            phases_json TEXT,
            equity_curve_json TEXT,
            detailed_trades_json TEXT
        )
    ''')

    # ============================================
    # GENERATED STRATEGIES TABLE
    # ============================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT UNIQUE NOT NULL,
            strategy_type TEXT,
            template_id TEXT,
            timeframe TEXT,
            direction TEXT,
            strategy_spec TEXT,
            python_code TEXT,
            created_at TEXT,
            batch_id TEXT
        )
    ''')

    # ============================================
    # TOP 5 STRATEGIES TABLE - FIXED FOR ACCUMULATION
    # ============================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS top5_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            original_strategy_id TEXT,
            batch_id TEXT NOT NULL,
            strategy_type TEXT,
            template_id TEXT,
            strategy_spec TEXT,
            batch_metrics TEXT,
            best_symbol TEXT,
            best_sharpe REAL,
            rank_in_batch INTEGER,
            validation_status TEXT DEFAULT 'pending',
            validation_result TEXT,
            created_at TEXT,
            validated_at TEXT,
            UNIQUE(batch_id, original_strategy_id)
        )
    ''')

    # ============================================
    # BATCH HISTORY TABLE
    # ============================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT UNIQUE NOT NULL,
            total_strategies INTEGER,
            total_symbols INTEGER,
            completed_validations INTEGER,
            start_date TEXT,
            end_date TEXT,
            config_json TEXT,
            created_at TEXT,
            completed_at TEXT,
            status TEXT DEFAULT 'running'
        )
    ''')

    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_validations_decision ON validations(decision)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_validations_symbol ON validations(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_top5_batch ON top5_strategies(batch_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_top5_sharpe ON top5_strategies(best_sharpe DESC)')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


# ============================================
# VALIDATION FUNCTIONS
# ============================================

def save_validation(validation_id: str, request: Dict, report: Dict):
    """Save validation to database"""
    conn = get_connection()
    cursor = conn.cursor()

    phases = report.get('phases', {})

    try:
        cursor.execute('''
            INSERT OR REPLACE INTO validations (
                validation_id, strategy_name, strategy_code, symbol,
                start_date, end_date, initial_capital, position_size,
                commission, slippage, spread, created_at,
                final_score, grade, decision, confidence_level, risk_level,
                total_return, sharpe_ratio, max_drawdown, win_rate,
                total_trades, total_costs, profit_factor, annual_return,
                phase1_score, phase1_passed, phase2_score, phase2_passed,
                phase3_score, phase3_passed, phase4_score, phase4_passed,
                phase5_score, phase5_passed, phase6_score, phase6_passed,
                phase7_score, phase7_passed,
                phases_json, equity_curve_json, detailed_trades_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            validation_id,
            request.get('strategy_name', ''),
            request.get('strategy_code', ''),
            request.get('symbol', ''),
            request.get('start_date', ''),
            request.get('end_date', ''),
            request.get('initial_capital', 100000),
            request.get('position_size', 0.95),
            request.get('commission', 0.001),
            request.get('slippage', 0.0005),
            request.get('spread', 0.0001),
            datetime.now().isoformat(),
            report.get('final_score', 0),
            report.get('grade', 'F'),
            report.get('decision', 'REJECTED'),
            report.get('confidence_level', 'NONE'),
            report.get('risk_level', 'CRITICAL'),
            report.get('total_return', 0),
            report.get('sharpe_ratio', 0),
            report.get('max_drawdown', 0),
            report.get('win_rate', 0),
            report.get('total_trades', 0),
            report.get('total_costs', 0),
            report.get('profit_factor', 0),
            report.get('annual_return', 0),
            phases.get('phase1', {}).get('score', 0),
            1 if phases.get('phase1', {}).get('passed', False) else 0,
            phases.get('phase2', {}).get('score', 0),
            1 if phases.get('phase2', {}).get('passed', False) else 0,
            phases.get('phase3', {}).get('score', 0),
            1 if phases.get('phase3', {}).get('passed', False) else 0,
            phases.get('phase4', {}).get('score', 0),
            1 if phases.get('phase4', {}).get('passed', False) else 0,
            phases.get('phase5', {}).get('score', 0),
            1 if phases.get('phase5', {}).get('passed', False) else 0,
            phases.get('phase6', {}).get('score', 0),
            1 if phases.get('phase6', {}).get('passed', False) else 0,
            phases.get('phase7', {}).get('score', 0),
            1 if phases.get('phase7', {}).get('passed', False) else 0,
            json.dumps(phases),
            json.dumps(report.get('equity_curve', {})),
            json.dumps(report.get('detailed_trades', []))
        ))
        conn.commit()
        logger.info(f"Validation {validation_id} saved successfully")
    except Exception as e:
        logger.error(f"Error saving validation: {e}")
        raise
    finally:
        conn.close()


def get_all_validations() -> List[Dict]:
    """Get all validations ordered by date"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM validations ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_validation(validation_id: str) -> Optional[Dict]:
    """Get single validation by ID with parsed JSON fields"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM validations WHERE validation_id = ?', (validation_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        result = dict(row)
        result['phases'] = json.loads(result.get('phases_json', '{}'))
        result['equity_curve'] = json.loads(result.get('equity_curve_json', '{}'))
        result['detailed_trades'] = json.loads(result.get('detailed_trades_json', '[]'))
        return result
    return None


def delete_validation(validation_id: str) -> bool:
    """Delete validation by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM validations WHERE validation_id = ?', (validation_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def delete_all_validations() -> int:
    """Delete all validations"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM validations')
    count = cursor.fetchone()['count']
    cursor.execute('DELETE FROM validations')
    conn.commit()
    conn.close()
    return count


def delete_rejected_validations() -> int:
    """Delete only rejected validations"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM validations WHERE decision = 'REJECTED'")
    count = cursor.fetchone()['count']
    cursor.execute("DELETE FROM validations WHERE decision = 'REJECTED'")
    conn.commit()
    conn.close()
    return count


def get_statistics() -> Dict:
    """Get overall validation statistics"""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute('SELECT COUNT(*) as total FROM validations')
    stats['total_validations'] = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as approved FROM validations WHERE decision = 'APPROVED'")
    stats['approved'] = cursor.fetchone()['approved']

    cursor.execute("SELECT COUNT(*) as conditional FROM validations WHERE decision = 'CONDITIONAL'")
    stats['conditional'] = cursor.fetchone()['conditional']

    cursor.execute("SELECT COUNT(*) as rejected FROM validations WHERE decision = 'REJECTED'")
    stats['rejected'] = cursor.fetchone()['rejected']

    cursor.execute('SELECT AVG(final_score) as avg_score FROM validations')
    avg = cursor.fetchone()['avg_score']
    stats['avg_score'] = round(avg, 1) if avg else 0

    cursor.execute('''
        SELECT symbol, COUNT(*) as count
        FROM validations
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 5
    ''')
    stats['top_symbols'] = [{'symbol': row['symbol'], 'count': row['count']} for row in cursor.fetchall()]

    conn.close()
    return stats


# ============================================
# VALIDATION RESULT FUNCTIONS (for 7-Phase)
# ============================================

def save_validation_result(validation_data: Dict):
    """Save a validation result from the 7-phase pipeline"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO validations (
            validation_id, strategy_name, created_at,
            final_score, decision, phases_json
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        validation_data.get('validation_id', ''),
        validation_data.get('strategy_id', ''),
        validation_data.get('created_at', datetime.now().isoformat()),
        validation_data.get('overall_score', 0),
        'APPROVED' if validation_data.get('overall_passed') else 'REJECTED',
        json.dumps(validation_data.get('phase_results', []))
    ))

    conn.commit()
    conn.close()
    logger.info(f"Saved validation result {validation_data.get('validation_id')}")


def get_validation_results(strategy_id: str) -> List[Dict]:
    """Get all validation results for a strategy"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM validations
        WHERE strategy_name = ?
        ORDER BY created_at DESC
    ''', (strategy_id,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        result = dict(row)
        if result.get('phases_json'):
            result['phase_results'] = json.loads(result['phases_json'])
        results.append(result)

    return results


# ============================================
# GENERATED STRATEGIES FUNCTIONS
# ============================================

def save_generated_strategies(strategies: List[Dict], batch_id: str = None):
    """Save generated strategies to database"""
    conn = get_connection()
    cursor = conn.cursor()

    if batch_id is None:
        batch_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Clear previous generated strategies (they're temporary)
    cursor.execute('DELETE FROM generated_strategies')

    for s in strategies:
        strategy_id = s.get('id', str(uuid.uuid4())[:8])
        cursor.execute('''
            INSERT INTO generated_strategies
            (strategy_id, strategy_type, template_id, timeframe, direction, strategy_spec, created_at, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            s.get('strategy_type', ''),
            s.get('template_id', ''),
            s.get('timeframe', ''),
            s.get('entry_logic', {}).get('direction', 'long'),
            json.dumps(s),
            datetime.now().isoformat(),
            batch_id
        ))

    conn.commit()
    conn.close()
    logger.info(f"Saved {len(strategies)} generated strategies (batch: {batch_id})")


def get_generated_strategies(limit: int = None, offset: int = 0, batch_id: str = None) -> List[Dict]:
    """Load generated strategies from database"""
    conn = get_connection()
    cursor = conn.cursor()

    query = 'SELECT strategy_spec FROM generated_strategies'
    params = []

    if batch_id:
        query += ' WHERE batch_id = ?'
        params.append(batch_id)

    query += ' ORDER BY id'

    if limit:
        query += ' LIMIT ? OFFSET ?'
        params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [json.loads(row['strategy_spec']) for row in rows]


def get_strategy_by_id(strategy_id: str) -> Optional[Dict]:
    """Get a single strategy by ID from generated or top5 tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # Try generated strategies first
    cursor.execute(
        'SELECT strategy_spec FROM generated_strategies WHERE strategy_id = ?',
        (strategy_id,)
    )
    row = cursor.fetchone()

    if row:
        conn.close()
        return json.loads(row['strategy_spec'])

    # Try top5 strategies
    cursor.execute(
        'SELECT strategy_spec FROM top5_strategies WHERE strategy_id = ? OR original_strategy_id = ?',
        (strategy_id, strategy_id)
    )
    row = cursor.fetchone()

    conn.close()

    if row:
        return json.loads(row['strategy_spec'])

    return None


def clear_generated_strategies():
    """Clear all generated strategies"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM generated_strategies')
    conn.commit()
    conn.close()
    logger.info("Cleared all generated strategies")


# ============================================
# TOP 5 STRATEGIES FUNCTIONS - FIXED FOR ACCUMULATION
# ============================================

def save_top5_strategies(strategies: List[Dict], batch_id: str = None) -> str:
    """
    Save top 5 strategies to database - ACCUMULATES instead of replacing.

    IMPORTANT FIX: Each batch gets a unique batch_id, and strategies are
    stored with this batch_id. This allows multiple batches of Top 5 to
    coexist in the database.

    Returns: batch_id used for this save operation
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Generate unique batch_id if not provided
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    logger.info(f"Saving {len(strategies)} Top 5 strategies with batch_id: {batch_id}")

    for rank, s in enumerate(strategies, 1):
        original_id = s.get('id', str(uuid.uuid4())[:8])
        # Create unique strategy_id combining batch and original
        unique_strategy_id = f"{batch_id}_{original_id}"

        batch_metrics = s.get('batch_metrics', {})

        try:
            cursor.execute('''
                INSERT INTO top5_strategies
                (strategy_id, original_strategy_id, batch_id, strategy_type, template_id,
                 strategy_spec, batch_metrics, best_symbol, best_sharpe, rank_in_batch, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                unique_strategy_id,
                original_id,
                batch_id,
                s.get('strategy_type', ''),
                s.get('template_id', ''),
                json.dumps(s),
                json.dumps(batch_metrics),
                batch_metrics.get('best_symbol', ''),
                batch_metrics.get('best_sharpe', 0),
                rank,
                datetime.now().isoformat()
            ))
        except sqlite3.IntegrityError:
            # Strategy already exists in this batch, update it
            cursor.execute('''
                UPDATE top5_strategies
                SET strategy_spec = ?, batch_metrics = ?, best_symbol = ?, best_sharpe = ?, rank_in_batch = ?
                WHERE batch_id = ? AND original_strategy_id = ?
            ''', (
                json.dumps(s),
                json.dumps(batch_metrics),
                batch_metrics.get('best_symbol', ''),
                batch_metrics.get('best_sharpe', 0),
                rank,
                batch_id,
                original_id
            ))

    conn.commit()
    conn.close()

    logger.info(f"Successfully saved Top 5 strategies for batch {batch_id}")
    return batch_id


def get_top5_strategies(limit: int = None, batch_id: str = None) -> List[Dict]:
    """
    Get top 5 strategies from database.

    Args:
        limit: Maximum number of strategies to return (None = all)
        batch_id: Filter by specific batch (None = all batches)

    Returns strategies ordered by best_sharpe descending.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = 'SELECT * FROM top5_strategies'
    params = []

    if batch_id:
        query += ' WHERE batch_id = ?'
        params.append(batch_id)

    query += ' ORDER BY best_sharpe DESC'

    if limit:
        query += ' LIMIT ?'
        params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        strategy = json.loads(row['strategy_spec'])
        strategy['db_id'] = row['id']
        strategy['batch_id'] = row['batch_id']
        strategy['rank_in_batch'] = row['rank_in_batch']
        strategy['validation_status'] = row['validation_status']
        if row['validation_result']:
            strategy['validation_result'] = json.loads(row['validation_result'])
        if row['batch_metrics']:
            strategy['batch_metrics'] = json.loads(row['batch_metrics'])
        results.append(strategy)

    return results


def get_top5_batches() -> List[Dict]:
    """Get list of all Top 5 batches with summary stats"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            batch_id,
            COUNT(*) as strategy_count,
            MAX(best_sharpe) as max_sharpe,
            AVG(best_sharpe) as avg_sharpe,
            MIN(created_at) as created_at
        FROM top5_strategies
        GROUP BY batch_id
        ORDER BY created_at DESC
    ''')

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def update_top5_validation_result(strategy_id: str, validation_result: Dict):
    """Update validation result for a Top 5 strategy"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE top5_strategies
        SET validation_status = ?, validation_result = ?, validated_at = ?
        WHERE strategy_id = ?
    ''', (
        'validated',
        json.dumps(validation_result),
        datetime.now().isoformat(),
        strategy_id
    ))

    conn.commit()
    conn.close()
    logger.info(f"Updated validation result for strategy {strategy_id}")


def delete_top5_strategy(strategy_id: str = None, batch_id: str = None, db_id: int = None):
    """
    Delete Top 5 strategy by various identifiers.

    Args:
        strategy_id: The unique strategy_id
        batch_id: Delete all strategies from a batch
        db_id: The database primary key id
    """
    conn = get_connection()
    cursor = conn.cursor()

    if db_id:
        cursor.execute('DELETE FROM top5_strategies WHERE id = ?', (db_id,))
    elif strategy_id:
        cursor.execute('DELETE FROM top5_strategies WHERE strategy_id = ?', (strategy_id,))
    elif batch_id:
        cursor.execute('DELETE FROM top5_strategies WHERE batch_id = ?', (batch_id,))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    logger.info(f"Deleted {deleted} Top 5 strategies")
    return deleted


def clear_top5_strategies() -> int:
    """Clear all top 5 strategies"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM top5_strategies')
    count = cursor.fetchone()['count']
    cursor.execute('DELETE FROM top5_strategies')
    conn.commit()
    conn.close()
    logger.info(f"Cleared {count} Top 5 strategies")
    return count


def get_top5_history() -> List[Dict]:
    """Get history of all Top 5 batches"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            batch_id,
            COUNT(*) as strategy_count,
            MAX(best_sharpe) as max_sharpe,
            AVG(best_sharpe) as avg_sharpe,
            MIN(created_at) as created_at
        FROM top5_strategies
        GROUP BY batch_id
        ORDER BY created_at DESC
    ''')

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ============================================
# BATCH HISTORY FUNCTIONS
# ============================================

def save_batch_history(batch_id: str, config: Dict, total_strategies: int, total_symbols: int):
    """Save batch validation history"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO batch_history
        (batch_id, total_strategies, total_symbols, completed_validations, config_json, created_at, status)
        VALUES (?, ?, ?, 0, ?, ?, 'running')
    ''', (
        batch_id,
        total_strategies,
        total_symbols,
        json.dumps(config),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def update_batch_history(batch_id: str, completed: int = None, status: str = None):
    """Update batch history progress"""
    conn = get_connection()
    cursor = conn.cursor()

    updates = []
    params = []

    if completed is not None:
        updates.append('completed_validations = ?')
        params.append(completed)

    if status is not None:
        updates.append('status = ?')
        params.append(status)
        if status == 'completed':
            updates.append('completed_at = ?')
            params.append(datetime.now().isoformat())

    if updates:
        params.append(batch_id)
        cursor.execute(f'''
            UPDATE batch_history SET {', '.join(updates)} WHERE batch_id = ?
        ''', params)
        conn.commit()

    conn.close()


def get_batch_history(limit: int = 20) -> List[Dict]:
    """Get batch validation history"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM batch_history
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize database on import
init_database()
